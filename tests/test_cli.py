# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from fp4local_abliterated.cli import main


def _base(tmp_path, hidden_size=8):
    base = tmp_path / "base"
    (base / "tensors").mkdir(parents=True)
    (base / "tensors" / "shard.safetensors").write_bytes(b"weights")
    (base / ".cache").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "linked.json").write_text("{}")
    (base / "linked").symlink_to(tmp_path / "elsewhere")
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"hidden_size": hidden_size},
    }
    (base / "config.json").write_text(json.dumps(config))
    return base, config


def test_writes_config_block_and_links_the_rest(tmp_path):
    base, config = _base(tmp_path)
    direction = tmp_path / "v.pt"
    torch.save(torch.full((8,), 2.0), direction)

    main([str(base), str(tmp_path / "out"), "--direction", str(direction), "--lambda", "1.5"])

    out = tmp_path / "out"
    written = json.loads((out / "config.json").read_text())
    block = written.pop("abliteration_config")
    assert written == config
    assert block["lambda"] == 1.5
    assert block["direction"] == pytest.approx([8**-0.5] * 8)
    shard = out / "tensors" / "shard.safetensors"
    assert shard.samefile(base / "tensors" / "shard.safetensors")
    assert (out / "linked" / "linked.json").samefile(tmp_path / "elsewhere" / "linked.json")
    assert not (out / ".cache").exists()


@pytest.mark.parametrize(
    "values, error", [(torch.ones(4), "4 values"), (torch.zeros(8), "norm is 0")]
)
def test_rejects_an_unusable_direction(tmp_path, values, error):
    base, _ = _base(tmp_path)
    direction = tmp_path / "v.pt"
    torch.save(values, direction)

    with pytest.raises(SystemExit, match=error):
        main([str(base), str(tmp_path / "out"), "--direction", str(direction)])
