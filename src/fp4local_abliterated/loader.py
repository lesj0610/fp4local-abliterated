# SPDX-License-Identifier: Apache-2.0
"""Teach vLLM's default loader the FP4-Local checkpoint layout.

FP4-Local keeps some or all of its safetensors shards under ``tensors/``
(listed by the root index) and stores expert block scales CSF-compressed.
vLLM pull requests #64739 and #64312 add both. The shard lookup changes
nothing once vLLM finds those shards itself, and the CSF decoding is skipped
once vLLM has its own decoders.
"""

import importlib.util
import json
import os

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.logger import init_logger

logger = init_logger(f"vllm.{__name__}")

_NO_WEIGHTS = "Cannot find any model weights"


def _indexed_safetensors_files(hf_folder: str, index_file: str) -> list[str]:
    index_path = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_path):
        return []
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    files = (os.path.join(hf_folder, name) for name in sorted(set(weight_map.values())))
    return [f for f in files if os.path.isfile(f)]


def patch_default_loader() -> None:
    from vllm.model_executor.model_loader import default_loader as dl
    from vllm.model_executor.model_loader.weight_cache import protocol

    loader_cls = dl.DefaultModelLoader
    if getattr(loader_cls, "_fp4local_patched", False):
        return
    filter_files = dl.filter_duplicate_safetensors_files
    prepare_weights = loader_cls._prepare_weights

    def filter_duplicate_safetensors_files(hf_weights_files, hf_folder, index_file):
        # Add the shards the index lists in a subdirectory; vLLM's filter then
        # keeps the listed files and fails on any that are missing.
        listed = _indexed_safetensors_files(hf_folder, index_file)
        files = hf_weights_files + [f for f in listed if f not in hf_weights_files]
        return filter_files(files, hf_folder, index_file)

    def _prepare_weights(
        self,
        model_name_or_path,
        subfolder,
        revision,
        fall_back_to_pt,
        allow_patterns_overrides,
    ):
        try:
            return prepare_weights(
                self,
                model_name_or_path,
                subfolder,
                revision,
                fall_back_to_pt,
                allow_patterns_overrides,
            )
        except RuntimeError as e:
            if allow_patterns_overrides is not None or _NO_WEIGHTS not in str(e):
                raise
            # Shards may live in a subdirectory named by the index.
            model = (
                dl.maybe_download_from_modelscope(model_name_or_path, revision)
                or model_name_or_path
            )
            if os.path.isdir(model):
                hf_folder = model
            else:
                hf_folder = dl.download_weights_from_hf(
                    model,
                    self.load_config.download_dir,
                    ["*.safetensors"],
                    revision,
                    subfolder=subfolder,
                    ignore_patterns=self.load_config.ignore_patterns,
                )
                dl.download_safetensors_index_file_from_hf(
                    model,
                    SAFE_WEIGHTS_INDEX_NAME,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            if subfolder is not None:
                hf_folder = os.path.join(hf_folder, subfolder)
            files = filter_duplicate_safetensors_files(
                [], hf_folder, SAFE_WEIGHTS_INDEX_NAME
            )
            if not files:
                raise
            return hf_folder, files, True, SAFE_WEIGHTS_INDEX_NAME

    # The IPC weight cache keys on the same files.
    dl.filter_duplicate_safetensors_files = filter_duplicate_safetensors_files
    protocol.filter_duplicate_safetensors_files = filter_duplicate_safetensors_files
    loader_cls._prepare_weights = _prepare_weights
    if not importlib.util.find_spec(
        "vllm.model_executor.layers.quantization.utils.csf_scales"
    ):
        from .csf_scales import decode_csf_scale_streams

        get_weights_iterator = loader_cls._get_weights_iterator

        def _get_weights_iterator(self, source):
            return decode_csf_scale_streams(get_weights_iterator(self, source))

        loader_cls._get_weights_iterator = _get_weights_iterator
    loader_cls._fp4local_patched = True
    logger.debug("Patched DefaultModelLoader for subdirectory shards and CSF scales")
