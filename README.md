# fp4local-abliterated

A vLLM plugin that removes a refusal direction at inference time, so quantized checkpoints can be abliterated without touching (or re-quantizing) their weights. It was written for [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), whose QAD-trained NVFP4 and MXFP8 weights would lose accuracy if they were dequantized, edited and quantized again.

At each residual-stream writer listed for the architecture the plugin replaces the output `x` with `x − λ (vᵀx) v`, computed in fp32. That is the same as editing each writer to `W − λ v vᵀ W`, which is what weight-editing abliteration does. Residual writes outside that list are left as they are; for Qwen4Exp these are the PLE short-convolution output and the vision tower's embeddings and deepstack inputs.

The plugin also loads FP4-Local's CSF-compressed NVFP4 scales and its `tensors/` shard layout, which vLLM releases do not read yet. That part is adapted from vLLM pull request #60646, wraps two internal methods of vLLM's default loader, and switches itself off once vLLM ships the feature.

## Requirements

vLLM 0.32.1rc0 or newer, declared as a dependency. FP4-Local stores its PLE table in NVFP4, which vLLM reads since vllm-project/vllm#56273. vLLM 0.32.0 predates that change and allocates the table as bf16, 95 GiB instead of 27 GiB; with `--engram-config '{"cpu_offload": true}'` that is pinned host memory, which can exhaust the host. Until 0.32.1 is on PyPI, install a nightly build:

```bash
uv pip install -U vllm --pre --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

For a custom vLLM build whose version string does not satisfy the requirement, install the plugin with `--no-deps`; the build must include #56273.

## Making an abliterated checkpoint

The direction is an input; this package does not compute it. Given one (a 1-D tensor of `hidden_size` values in a `.pt` or `.safetensors` file):

```bash
pip install fp4local-abliterated
fp4local-abliterate <base checkpoint> <output dir> --direction v.pt --lambda 1.5 \
  --source "where the direction came from"
```

The output directory holds the base files (hard-linked when on the same filesystem) and a `config.json` with one added block, the same way quantizers add `quantization_config`:

```json
"abliteration_config": {"lambda": 1.5, "direction": [...], "source": "..."}
```

`direction` is stored normalized. λ = 1 removes the component along it; λ > 1 reverses part of it. The MTP draft model reads the same block, so target and draft stay consistent. To try another λ, write a new checkpoint; `--hf-overrides` only reaches the target model, not the draft.

Do not add the block to a checkpoint whose weights were already abliterated; the projection would be applied twice.

## Serving

```bash
vllm serve <output dir> \
  --engram-config '{"cpu_offload": true}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

vLLM loads the plugin through its `vllm.general_plugins` entry point; no flag is needed. Checkpoints without `abliteration_config` run unchanged.

`--engram-config` is optional. It moves the PLE table from GPU memory into pinned host memory, which cannot be swapped out: plan for 36 GiB of free host RAM (the 27 GiB table, rounded up to powers of two by PyTorch's pinned-memory allocator).

With `--tensor-parallel-size` above 1, add `--mm-encoder-tp-mode data`. The vision MLP's intermediate size (4304) cannot be split across GPUs into NVFP4 blocks of 16, so each GPU keeps its own copy of the vision tower instead.

At startup each worker logs one line for the model and one for the MTP draft:

```
Abliteration on AbliteratedQwen4ExpForConditionalGeneration: lambda 1.5, ... residual writes in ... layers
Abliteration on AbliteratedQwen4ExpMTP: lambda 1.5, 5 residual writes in 1 layers
```

If the line is missing, the model is being served without abliteration.

## Architectures

| vLLM architecture | Models | Tested with |
|---|---|---|
| Qwen4Exp (`Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration`, `Qwen4ExpMTP`) | Qwen3.8-Flash-Next family | `nvidia/Qwen3.8-Flash-Next-FP4-Local` on vLLM 0.32.1rc1.dev467 (nightly), sm_80, tensor parallel, MTP on |

Each architecture is one entry in [`adapters.py`](src/fp4local_abliterated/adapters.py): the vLLM module that defines it, the decoder layer class, and which submodules write into the residual stream. Everything else is shared. The plugin registers an `Abliterated<Arch>` subclass for every listed architecture through `ModelRegistry`, and attaches forward hooks after vLLM builds the model.

To add an architecture, list its residual writers (embedding, attention output, MLP or MoE output, and anything else added to the residual) as they are named in vLLM's model code, then check that the startup log counts one write per writer per layer.

## Where the direction for FP4-Local came from

The direction used for [lesj0610/Qwen3.8-Flash-Next-FP4-Local-Abliterated](https://huggingface.co/lesj0610/Qwen3.8-Flash-Next-FP4-Local-Abliterated) was not captured by us. It is the direction of [windowsxp811203/Qwen3.8-Flash-Next-Abliterated](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated), a bf16 weight-edited build of the same base model, recovered exactly from the difference between its edited and original weights (the difference is rank-1). Their model card describes how it was captured: difference of mean residual activations on harmful and harmless prompts, after 37 of the 48 decoder layers. The method is from Arditi et al., [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (2024).

## Tests

```bash
pytest tests/
```

The CSF and CLI tests run on CPU. The projection tests need a CUDA GPU.

## License

Apache-2.0. `csf_scales.py` is adapted from vLLM (Apache-2.0).
