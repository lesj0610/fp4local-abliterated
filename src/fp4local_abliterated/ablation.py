# SPDX-License-Identifier: Apache-2.0
"""Inference-time abliteration.

Every write into the residual stream has its component along the refusal
direction ``v`` scaled by ``1 - lam``: ``x - lam * (v . x) v``. This equals
editing those weights to ``W - lam * v v^T W`` but leaves the quantized weights
untouched. The writes are module outputs, so forward hooks apply it without
touching vLLM's model code; adapters.py says which modules they are.

The checkpoint enables it with a top-level ``abliteration_config`` block in
config.json (``direction``: hidden_size floats, ``lambda``), written by the
``fp4local-abliterate`` CLI.
"""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from .adapters import Adapter

logger = init_logger(f"vllm.{__name__}")


@triton.jit
def _ablate_kernel(
    x_ptr,
    out_ptr,
    direction_ptr,
    x_stride,
    out_stride,
    lam,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    direction = tl.load(direction_ptr + offs, mask=mask, other=0.0)
    coef = tl.sum(x * direction) * lam
    out = x - coef * direction
    tl.store(
        out_ptr + row * out_stride + offs, out.to(out_ptr.dtype.element_ty), mask=mask
    )


def ablate(
    x: torch.Tensor, direction: torch.Tensor, lam: float, inplace: bool = False
) -> torch.Tensor:
    """Project ``x`` along its last dim in fp32; rows may be strided views."""
    if x.shape[-1] != direction.numel():
        raise ValueError(
            f"Abliteration needs full hidden rows ({direction.numel()}), got "
            f"{tuple(x.shape)}; tensor-parallel shards are not supported here"
        )
    rows = x.reshape(-1, x.shape[-1])
    if rows.stride(-1) != 1:
        if inplace:
            raise ValueError("In-place abliteration needs unit-stride rows")
        rows = rows.contiguous()
    out = rows if inplace else torch.empty(rows.shape, dtype=x.dtype, device=x.device)
    _ablate_kernel[(rows.shape[0],)](
        rows,
        out,
        direction,
        rows.stride(0),
        out.stride(0),
        lam,
        N=rows.shape[1],
        BLOCK=triton.next_power_of_2(rows.shape[1]),
    )
    return x if inplace else out.view(x.shape)


def _hook(owner: nn.Module, lam: float, packed: bool):
    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        direction = owner.ablit_direction
        if packed:
            ablate(out[..., -direction.numel() :], direction, lam, inplace=True)
            return output
        out = ablate(out, direction, lam)
        return (out, *output[1:]) if isinstance(output, tuple) else out

    return hook


def attach_abliteration(model: nn.Module, hf_config, adapter: Adapter) -> None:
    config = getattr(hf_config, "abliteration_config", None)
    if config is None:
        return
    lam = float(config["lambda"])
    direction = torch.tensor(config["direction"], dtype=torch.float32)
    model.register_buffer(
        "ablit_direction", direction / direction.norm(), persistent=False
    )

    sites = 0
    layers = 0
    for module in model.modules():
        kind = type(module).__name__
        layers += kind == adapter.layer_class
        for name in adapter.writers.get(kind, ()):
            target = getattr(module, name, None)
            if target is None or type(target).__name__ == "PPMissingLayer":
                continue
            packed = (kind, name) in adapter.packed_writers
            target.register_forward_hook(_hook(model, lam, packed))
            sites += 1

    if layers == 0:
        raise RuntimeError(
            f"{type(model).__name__} has an abliteration_config but no "
            f"{adapter.layer_class} was found; this vLLM version is not supported"
        )
    logger.info(
        "Abliteration on %s: lambda %.4g, %d residual writes in %d layers",
        type(model).__name__,
        lam,
        sites,
        layers,
    )
