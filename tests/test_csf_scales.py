# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from fp4local_abliterated.csf_scales import decode_csf_scale, decode_csf_scale_streams


def _encode(scales: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reference csf encoder: row base = row min (capped at 240), outliers as
    exceptions."""
    raw = scales.view(torch.uint8).to(torch.int64)
    rows, cols = raw.shape
    bases = raw.min(dim=1, keepdim=True).values.clamp(max=240)
    codes = raw - bases
    outlier = (codes < 0) | (codes > 15)
    codes = torch.where(outlier, 0, codes)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    fixed = torch.cat(
        (
            bases.to(torch.uint8).reshape(rows // 16, 16),
            packed.reshape(rows // 16, -1),
        ),
        dim=1,
    )
    positions = outlier.reshape(-1).nonzero().reshape(-1)
    words = (raw.reshape(-1)[positions] << 24) | positions
    return {
        "nvfp4_csf_fixed": fixed,
        "nvfp4_csf_exceptions": words.to(torch.int32).view(torch.uint32),
    }


def _encode_32x2(raw: np.ndarray) -> dict[str, torch.Tensor]:
    """Reference csf_32x2_v1 encoder, written as the format's loops."""
    rows, cols = raw.shape
    bases = raw.min(axis=1).clip(max=240)
    codes: list[int] = []
    origins: list[int] = []
    offsets: list[int] = []
    words: list[int] = []
    group = 0
    for row in range(0, rows, 128):
        for col in range(0, cols, 4):
            if group % 32 == 0:
                origins.append(2 * len(words))
                offsets.append(0)
                page = len(words)
            for pair in range(col, min(col + 4, cols), 2):
                for block in range(row, min(row + 128, rows), 32):
                    for i in range(32):
                        for j in range(min(2, cols - pair)):
                            value = int(raw[block + i, pair + j])
                            code = value - int(bases[block + i])
                            codes.append(code if 0 <= code <= 15 else 0)
                            if not 0 <= code <= 15:
                                words.append(value << 8 | 2 * i + j)
                    offsets.append(len(words) - page)
            group += 1
    nibbles = np.array(codes, dtype=np.uint8)
    streams = (
        ("bases", bases, np.uint8),
        ("codes", nibbles[0::2] | nibbles[1::2] << 4, np.uint8),
        ("page_bases", origins, np.uint32),
        ("offsets", offsets, np.uint16),
        ("exceptions", words, np.uint16),
    )
    return {
        f"csf32_{name}": torch.from_numpy(np.array(data, dtype=dtype))
        for name, data, dtype in streams
    }


def _raw_scales(shape: tuple[int, int]) -> torch.Tensor:
    torch.manual_seed(0)
    raw = torch.randint(0x30, 0x40, shape, dtype=torch.uint8)
    raw.view(-1)[torch.randperm(raw.numel())[:64]] = 0x7E  # out-of-window bytes
    return raw


@pytest.mark.parametrize("shape", [(640, 160), (2560, 40)])
def test_csf_decode_matches_source_scales(shape):
    raw = _raw_scales(shape)
    streams = _encode(raw.view(torch.float8_e4m3fn))
    assert streams["nvfp4_csf_exceptions"].numel() >= 64

    decoded = decode_csf_scale("csf", streams)

    assert decoded.dtype == torch.float8_e4m3fn
    assert torch.equal(decoded.view(torch.uint8), raw)


def test_csf_32x2_decode_matches_source_scales():
    # 352 rows leave a partial 128-row tile, 43 columns a 1-wide block, and
    # the 33 tiles span two pages.
    raw = _raw_scales((352, 43))
    raw[:32, :2] = 0x7E  # a block made only of exceptions
    streams = _encode_32x2(raw.numpy())

    decoded = decode_csf_scale("csf_32x2_v1", streams)

    assert torch.equal(decoded.view(torch.uint8), raw)


@pytest.mark.parametrize(
    "base, words", [(0xFF, []), (0x38, [5, 3]), (0x38, [3, 3]), (0x38, [64])]
)
def test_invalid_csf_streams_are_rejected(base, words):
    raw = torch.full((16, 4), 0x38, dtype=torch.uint8)
    streams = _encode(raw.view(torch.float8_e4m3fn))
    streams["nvfp4_csf_fixed"][0, 0] = base
    exceptions = torch.tensor(words, dtype=torch.int32).view(torch.uint32)
    streams["nvfp4_csf_exceptions"] = exceptions
    with pytest.raises(ValueError, match="CSF"):
        decode_csf_scale("csf", streams)


@pytest.mark.parametrize(
    "stream, index, value",
    [
        ("csf32_page_bases", 1, 0),  # page origin out of step
        ("csf32_offsets", 1, 65),  # block end past its 64 cells
        ("csf32_exceptions", 0, 0x7E40),  # position past the block
        ("csf32_exceptions", 1, 0x7E00),  # positions out of order
    ],
)
def test_invalid_csf_32x2_streams_are_rejected(stream, index, value):
    raw = torch.full((352, 43), 0x38, dtype=torch.uint8)
    raw[:32, :2] = 0x7E
    raw[128, 4] = 0x7E
    streams = _encode_32x2(raw.numpy())
    streams[stream].numpy()[index] = value
    with pytest.raises(ValueError, match="(?i)csf"):
        decode_csf_scale("csf_32x2_v1", streams)


@pytest.mark.parametrize("encoding", ["csf", "csf_32x2_v1"])
def test_streams_are_replaced_by_the_scale(encoding):
    raw = _raw_scales((352, 48))
    if encoding == "csf":
        streams = _encode(raw.view(torch.float8_e4m3fn))
    else:
        streams = _encode_32x2(raw.numpy())
    weights = [("e.weight", torch.zeros(1))]
    weights += [(f"e.weight_scale.{name}", t) for name, t in reversed(streams.items())]

    out = dict(decode_csf_scale_streams(weights))

    assert list(out) == ["e.weight", "e.weight_scale"]
    assert torch.equal(out["e.weight_scale"].view(torch.uint8), raw)


def test_unpaired_stream_is_rejected():
    fixed = _encode(torch.full((16, 4), 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="without a partner"):
        list(
            decode_csf_scale_streams(
                [("e.weight_scale.nvfp4_csf_fixed", fixed["nvfp4_csf_fixed"])]
            )
        )


@pytest.mark.parametrize(
    "shard_names",
    [
        ["tensors/model-00001-of-00001.safetensors"],
        # Root shards next to one under tensors/, like GLM-5.3-Flash-FP4-Local.
        [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "tensors/model-00003-of-00003.safetensors",
        ],
        ["pytorch_model.bin"],
    ],
)
def test_hub_snapshot_finds_shards_listed_by_the_index(
    tmp_path, monkeypatch, shard_names
):
    """Shards under tensors/ are found through the index, also by the IPC
    weight cache key, and a .bin-only snapshot never asks for the index."""
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import default_loader
    from vllm.model_executor.model_loader.weight_cache.protocol import hash_checkpoint

    from fp4local_abliterated.loader import patch_default_loader

    shards = [tmp_path / name for name in shard_names]
    for i, shard in enumerate(shards):
        shard.parent.mkdir(exist_ok=True)
        save_file({f"w{i}": torch.zeros(1)}, str(shard))
    fetched = []

    def fetch_index(model, index_file, cache_dir, subfolder, revision):
        fetched.append(index_file)
        weight_map = {f"w{i}": name for i, name in enumerate(shard_names)}
        (tmp_path / index_file).write_text(json.dumps({"weight_map": weight_map}))

    monkeypatch.setattr(
        default_loader, "download_weights_from_hf", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(
        default_loader, "download_safetensors_index_file_from_hf", fetch_index
    )
    patch_default_loader()
    loader = default_loader.DefaultModelLoader(LoadConfig(load_format="hf"))

    _, files, use_safetensors, _ = loader._prepare_weights(
        "org/model", None, None, fall_back_to_pt=False, allow_patterns_overrides=None
    )

    assert sorted(files) == sorted(map(str, shards))
    assert use_safetensors == bool(fetched) == shard_names[0].endswith(".safetensors")
    assert (hash_checkpoint(str(tmp_path)) is not None) == use_safetensors
    # A shard the index lists must not go missing quietly.
    if use_safetensors:
        shards[-1].unlink()
        with pytest.raises(FileNotFoundError, match="missing"):
            loader._prepare_weights(
                "org/model", None, None, False, allow_patterns_overrides=None
            )
