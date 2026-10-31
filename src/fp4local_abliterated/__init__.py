# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin for inference-time abliteration.

- Applies abliteration when config.json carries an ``abliteration_config``
  block (see the ``fp4local-abliterate`` CLI). Without it the model runs
  unchanged.
- Loads FP4-Local's CSF-compressed NVFP4 block scales (``csf`` and
  ``csf_32x2_v1``) and the shards its index lists under ``tensors/``, until
  vLLM does this itself.
"""

from .adapters import ADAPTERS


def register() -> None:
    from vllm import ModelRegistry

    from .loader import patch_default_loader
    from .models import PREFIX

    patch_default_loader()
    for adapter in ADAPTERS:
        for arch in adapter.architectures:
            ModelRegistry.register_model(
                arch, f"fp4local_abliterated.models:{PREFIX}{arch}"
            )
