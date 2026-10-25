# SPDX-License-Identifier: Apache-2.0
"""Inference-time abliteration for Qwen4Exp.

Every write into the residual stream (token embedding, attention and linear
attention output, MoE output, PLE value, MTP input projections) has its
component along the refusal direction ``v`` scaled by ``1 - lam``:
``x - lam * (v . x) v``. This equals editing those weights to
``W - lam * v v^T W`` but leaves the quantized weights untouched. The writes
are module outputs, so forward hooks apply it without touching vLLM's model
code.
"""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

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


def _hook_output(owner: nn.Module, lam: float):
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (ablate(output[0], owner.ablit_direction, lam), *output[1:])
        return ablate(output, owner.ablit_direction, lam)

    return hook


def _hook_ple_value(owner: nn.Module, lam: float, value_offset: int):
    # kv_proj returns [key | value]; only the value half reaches the residual.
    def hook(module, args, output):
        kv = output[0] if isinstance(output, tuple) else output
        ablate(kv[..., value_offset:], owner.ablit_direction, lam, inplace=True)
        return output

    return hook


def attach_abliteration(model: nn.Module, text_config) -> None:
    direction = getattr(text_config, "abliteration_direction", None)
    if direction is None:
        return
    lam = float(getattr(text_config, "abliteration_lambda", 1.0))
    direction = torch.tensor(direction, dtype=torch.float32)
    model.register_buffer(
        "ablit_direction", direction / direction.norm(), persistent=False
    )
    hook = _hook_output(model, lam)

    sites = 0
    layers = 0
    for module in model.modules():
        kind = type(module).__name__
        targets: list[nn.Module] = []
        if kind == "Qwen4ExpDecoderLayer":
            layers += 1
            for name in ("linear_attn", "self_attn", "mlp"):
                if hasattr(module, name):
                    targets.append(getattr(module, name))
        elif kind in ("Qwen4ExpModel", "Qwen4ExpMultiTokenPredictor"):
            for name in ("embed_tokens", "fc_embedding", "fc_hidden"):
                if hasattr(module, name):
                    targets.append(getattr(module, name))
        elif kind == "Qwen4ExpPLELayer":
            value_offset = module.kv_proj.output_sizes[0]
            module.kv_proj.register_forward_hook(
                _hook_ple_value(model, lam, value_offset)
            )
            sites += 1
        for target in targets:
            if type(target).__name__ == "PPMissingLayer":
                continue
            target.register_forward_hook(hook)
            sites += 1

    if layers == 0:
        raise RuntimeError(
            f"{type(model).__name__} has abliteration fields in its config but no "
            "Qwen4ExpDecoderLayer was found; this vLLM version is not supported"
        )
    logger.info(
        "Abliteration on %s: lambda %.4g, %d residual writes in %d layers",
        type(model).__name__,
        lam,
        sites,
        layers,
    )
