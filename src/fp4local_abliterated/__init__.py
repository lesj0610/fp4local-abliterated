# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin for Qwen3.8-Flash-Next FP4-Local checkpoints.

- Loads CSF-compressed NVFP4 block scales and shards kept in a subdirectory,
  until vLLM does this itself.
- Applies inference-time abliteration when the text config carries
  ``abliteration_direction`` and ``abliteration_lambda``. Without those fields
  the model runs unchanged.
"""

ARCHITECTURES = (
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
)


def register() -> None:
    from vllm import ModelRegistry

    from .loader import patch_default_loader

    patch_default_loader()
    for arch in ARCHITECTURES:
        ModelRegistry.register_model(
            arch, f"fp4local_abliterated.models:Abliterated{arch}"
        )
