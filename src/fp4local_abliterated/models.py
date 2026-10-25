# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp classes that attach abliteration after vLLM builds them.

``vllm.models.qwen4_exp`` resolves the platform's implementation, so these
subclass whatever vLLM would have used.
"""

from vllm.config import VllmConfig
from vllm.models import qwen4_exp

from .ablation import attach_abliteration


class AbliteratedQwen4ExpForCausalLM(qwen4_exp.Qwen4ExpForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        attach_abliteration(self, vllm_config.model_config.hf_text_config)


class AbliteratedQwen4ExpForConditionalGeneration(
    qwen4_exp.Qwen4ExpForConditionalGeneration
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        attach_abliteration(self, vllm_config.model_config.hf_text_config)


class AbliteratedQwen4ExpMTP(qwen4_exp.Qwen4ExpMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        attach_abliteration(self, vllm_config.model_config.hf_text_config)
