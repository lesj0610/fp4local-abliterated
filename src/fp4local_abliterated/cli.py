# SPDX-License-Identifier: Apache-2.0
"""Write an abliterated checkpoint: the base files, hard-linked where possible,
plus an ``abliteration_config`` block in config.json."""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from .adapters import find_adapter


def _load_direction(path: Path) -> torch.Tensor:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = list(load_file(path).values())
        if len(tensors) != 1:
            raise SystemExit(f"{path} holds {len(tensors)} tensors, expected one")
        return tensors[0]
    return torch.load(path, map_location="cpu", weights_only=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fp4local-abliterate", description=__doc__)
    parser.add_argument("base", type=Path, help="base checkpoint directory")
    parser.add_argument("out", type=Path, help="output directory, must not exist")
    parser.add_argument(
        "--direction",
        type=Path,
        required=True,
        help="refusal direction: one hidden_size tensor (.pt or .safetensors)",
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--source", help="where the direction came from")
    args = parser.parse_args(argv)

    config = json.loads((args.base / "config.json").read_text())
    architectures = config.get("architectures") or []
    if not any(find_adapter(a) for a in architectures):
        raise SystemExit(f"no adapter for {architectures}")
    hidden_size = config.get("text_config", config)["hidden_size"]
    direction = _load_direction(args.direction).float().flatten()
    if direction.numel() != hidden_size:
        raise SystemExit(f"direction has {direction.numel()} values, model {hidden_size}")
    norm = direction.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise SystemExit(f"direction norm is {norm.item()}, cannot normalize")
    block = {"lambda": args.lam, "direction": (direction / norm).tolist()}
    if args.source:
        block["source"] = args.source
    config["abliteration_config"] = block

    args.out.mkdir(parents=True)
    # followlinks: a checkpoint may symlink whole directories (e.g. tensors/).
    for root, dirs, files in os.walk(args.base, followlinks=True):
        rel = Path(root).relative_to(args.base)
        top = rel == Path(".")
        # .cache holds huggingface_hub download state, not checkpoint files.
        dirs[:] = sorted(d for d in dirs if not (top and d == ".cache"))
        (args.out / rel).mkdir(exist_ok=True)
        for name in sorted(files):
            if top and name == "config.json":
                continue
            src, dst = Path(root) / name, args.out / rel / name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
