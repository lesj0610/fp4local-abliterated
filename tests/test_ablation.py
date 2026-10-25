# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fp4local_abliterated.ablation import ablate, attach_abliteration
from fp4local_abliterated.adapters import QWEN4EXP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference(x: torch.Tensor, v: torch.Tensor, lam: float) -> torch.Tensor:
    xf = x.float()
    return (xf - lam * (xf @ v)[..., None] * v).to(x.dtype)


@pytest.mark.parametrize("shape", [(1, 2560), (7, 4, 2560)])
@pytest.mark.parametrize("lam", [1.0, 1.5])
def test_matches_fp32_projection(shape, lam):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(2560, device="cuda")
    v /= v.norm()

    out = ablate(x, v, lam)

    torch.testing.assert_close(out, _reference(x, v, lam), atol=0, rtol=1.6e-2)
    if lam == 1.0:
        assert (out.float() @ v).abs().max() < 0.05


def test_inplace_on_a_column_slice_leaves_the_rest():
    torch.manual_seed(0)
    kv = torch.randn(5, 3 * 2560, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(2560, device="cuda")
    v /= v.norm()
    expected = torch.cat([kv[:, :5120], _reference(kv[:, 5120:], v, 1.5)], dim=1)

    ablate(kv[:, 5120:], v, 1.5, inplace=True)

    torch.testing.assert_close(kv, expected, atol=0, rtol=1.6e-2)


def test_sharded_rows_are_rejected():
    x = torch.randn(3, 1280, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="full hidden rows"):
        ablate(x, torch.ones(2560, device="cuda"), 1.0)


class PPMissingLayer(nn.Identity):
    pass


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)


class Qwen4ExpPLELayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.kv_proj = nn.Linear(hidden, 3 * hidden, bias=False)


class Qwen4ExpModel(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embed_tokens = PPMissingLayer()
        self.layers = nn.ModuleList([Qwen4ExpDecoderLayer(hidden), Qwen4ExpPLELayer(hidden)])


def test_hooks_follow_the_adapter():
    torch.manual_seed(0)
    hidden = 64
    v = torch.randn(hidden)
    v /= v.norm()
    config = SimpleNamespace(abliteration_config={"direction": v.tolist(), "lambda": 1.0})
    with torch.device("cuda"):
        model = Qwen4ExpModel(hidden).to(torch.bfloat16)
        attach_abliteration(model, config, QWEN4EXP)
        x = torch.randn(5, hidden, dtype=torch.bfloat16)
    layer, ple = model.layers
    kv = ple.kv_proj(x)

    for write in (layer.self_attn(x), layer.mlp(x), kv[:, -hidden:]):
        assert (write.float() @ v.cuda()).abs().max() < 0.05
    assert torch.equal(kv[:, :-hidden], (x @ ple.kv_proj.weight.T)[:, :-hidden])
    assert torch.equal(model.embed_tokens(x), x)
