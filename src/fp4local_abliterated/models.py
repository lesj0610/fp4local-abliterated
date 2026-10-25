# SPDX-License-Identifier: Apache-2.0
"""``Abliterated<Arch>`` subclasses for every architecture in adapters.py.

They are built on first lookup, so vLLM only imports the model code it serves,
and they subclass whatever class vLLM itself resolves for the platform.
"""

import importlib

from .ablation import attach_abliteration
from .adapters import find_adapter

PREFIX = "Abliterated"


def __getattr__(name: str):
    adapter = find_adapter(name.removeprefix(PREFIX))
    if not name.startswith(PREFIX) or adapter is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    base = getattr(importlib.import_module(adapter.module), name.removeprefix(PREFIX))

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        base.__init__(self, vllm_config=vllm_config, prefix=prefix)
        attach_abliteration(self, vllm_config.model_config.hf_config, adapter)

    cls = type(name, (base,), {"__init__": __init__, "__module__": __name__})
    globals()[name] = cls
    return cls
