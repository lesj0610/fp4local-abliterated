# fp4local-abliterated

A vLLM plugin that serves [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) with its refusal direction removed at inference time. The quantized weights stay exactly as NVIDIA released them; the model's `config.json` carries the direction and strength, and the plugin projects every write into the residual stream.

The plugin also loads FP4-Local's CSF-compressed NVFP4 scales and its `tensors/` shard layout, which vLLM releases do not read yet. That part comes from vLLM pull request #60646 and switches itself off once vLLM ships it.

## Requirements

vLLM 0.32.x. The plugin wraps two internal methods of vLLM's default model loader and subclasses the Qwen4Exp model classes, so other vLLM versions may need an update here. vLLM is not a declared dependency, so installing the plugin never replaces the vLLM you already have.

## Usage

```bash
pip install git+https://github.com/lesj0610/fp4local-abliterated
vllm serve <abliterated checkpoint> \
  --engram-config '{"cpu_offload": true}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

vLLM picks the plugin up through its `vllm.general_plugins` entry point; no flag is needed. `--engram-config` keeps the 27 GiB PLE table in host memory and is optional. Models without the `abliteration_*` fields in their config run unchanged.

Check the server log at startup. Each worker prints a line like this for the model, and one for the MTP draft (`AbliteratedQwen4ExpMTP`):

```
Abliteration on AbliteratedQwen4ExpForConditionalGeneration: lambda 1.5, ... residual writes in ... layers
```

and loading prints `Decoded ... CSF expert scale tensors at load`. If the Abliteration line is missing, the model is being served without abliteration.

## Tests

```bash
pytest tests/
```

The CSF tests run on CPU. The projection kernel tests need a CUDA GPU.

## License

Apache-2.0. `csf_scales.py` is adapted from vLLM (Apache-2.0).
