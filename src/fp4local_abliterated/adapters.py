# SPDX-License-Identifier: Apache-2.0
"""Per-architecture description of the residual-stream writers.

Everything else (config parsing, the projection, hooking, registration) is
shared, so supporting another vLLM architecture means adding one entry here.
This module imports nothing from vLLM so the CLI can use it without a GPU.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Adapter:
    # vLLM module that defines the architectures below.
    module: str
    architectures: tuple[str, ...]
    # Decoder layer class; finding none means vLLM's model code has moved on.
    layer_class: str
    # Owner class name -> submodules whose output is added to the residual.
    writers: dict[str, tuple[str, ...]]
    # (owner, submodule) pairs whose output only ends with the residual write.
    packed_writers: frozenset[tuple[str, str]] = field(default_factory=frozenset)


QWEN4EXP = Adapter(
    module="vllm.models.qwen4_exp",
    architectures=(
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    ),
    layer_class="Qwen4ExpDecoderLayer",
    writers={
        "Qwen4ExpDecoderLayer": ("linear_attn", "self_attn", "mlp"),
        "Qwen4ExpModel": ("embed_tokens",),
        "Qwen4ExpMultiTokenPredictor": ("embed_tokens", "fc_embedding", "fc_hidden"),
        "Qwen4ExpPLELayer": ("kv_proj",),
    },
    # kv_proj packs [key | value]; only the value reaches the residual.
    packed_writers=frozenset({("Qwen4ExpPLELayer", "kv_proj")}),
)

ADAPTERS = (QWEN4EXP,)


def find_adapter(architecture: str) -> Adapter | None:
    return next((a for a in ADAPTERS if architecture in a.architectures), None)
