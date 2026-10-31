# fp4local-abliterated

A vLLM plugin for NVIDIA's FP4-Local checkpoints. It does two separate things:

- **Loading.** It reads the parts of the FP4-Local format that vLLM releases do not read yet: CSF-compressed NVFP4 expert scales and safetensors shards kept under `tensors/`. This works for any model vLLM serves, with or without abliteration.
- **Abliteration.** It removes a refusal direction at inference time, so a quantized checkpoint can be abliterated without touching (or re-quantizing) its weights. Quantization-aware trained weights, like the NVFP4 and MXFP8 weights of [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), would lose accuracy if they were dequantized, edited and quantized again. This works for the architectures listed under Architectures below.

## Requirements

vLLM 0.32.1rc0 or newer, declared as a dependency. Qwen3.8-Flash-Next-FP4-Local stores its PLE table in NVFP4, which vLLM reads since vLLM pull request #56273. vLLM 0.32.0 predates that change and allocates the table as bf16, 95 GiB instead of 27 GiB; with `--engram-config '{"cpu_offload": true}'` that is pinned host memory, which can exhaust the host. Until 0.32.1 is on PyPI, install a nightly build:

```bash
uv pip install -U vllm --pre --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
pip install fp4local-abliterated
```

For a custom vLLM build whose version string does not satisfy the requirement, install the plugin with `--no-deps`; the build must include #56273.

vLLM loads the plugin through its `vllm.general_plugins` entry point; no flag is needed.

## Loading FP4-Local checkpoints

| Checkpoint | Scale encoding | Shards |
|---|---|---|
| [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | `csf` | all under `tensors/` |
| [GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) | `csf_32x2_v1` | in the root, and one under `tensors/` |

The plugin decodes the scales while vLLM reads the weights, so the model gets the same scales as from an uncompressed checkpoint, and it adds the shards the index lists under `tensors/` to the ones vLLM finds in the root. A shard the index lists but that is missing fails the load.

The decoders give the same bytes as the reference decoders on every scale matrix of Qwen3.8-Flash-Next-FP4-Local and on 378 matrices covering all 42 layers of GLM-5.3-Flash-FP4-Local. Only Qwen3.8-Flash-Next-FP4-Local has been served with the plugin.

Both parts wrap internal functions of vLLM's default loader and follow vLLM pull requests #64312 (decoding) and #64739 (shards). The decoding switches itself off once vLLM has its own decoders, and the shard lookup changes nothing once vLLM finds those shards itself.

## Abliteration

At each residual-stream writer listed for the architecture the plugin replaces the output `x` with `x − λ (vᵀx) v`, computed in fp32. That is the same as editing each writer to `W − λ v vᵀ W`, which is what weight-editing abliteration does. Residual writes outside that list are left as they are; for Qwen4Exp these are the PLE short-convolution output and the vision tower's embeddings and deepstack inputs.

### Making an abliterated checkpoint

The direction is an input; this package does not compute it. Given one (a 1-D tensor of `hidden_size` values in a `.pt` or `.safetensors` file):

```bash
fp4local-abliterate <base checkpoint> <output dir> --direction v.pt --lambda 1.5 \
  --source "where the direction came from"
```

The output directory holds the base files (hard-linked when on the same filesystem) and a `config.json` with one added block, the same way quantizers add `quantization_config`:

```json
"abliteration_config": {"lambda": 1.5, "direction": [...], "source": "..."}
```

`direction` is stored normalized. λ = 1 removes the component along it; λ > 1 reverses part of it. An MTP draft model reads the same block, so target and draft stay consistent. To try another λ, write a new checkpoint; `--hf-overrides` only reaches the target model, not the draft.

Do not add the block to a checkpoint whose weights were already abliterated; the projection would be applied twice.

### Serving

Serve the output directory as you would serve the base checkpoint. Checkpoints without `abliteration_config` run unchanged.

At startup each worker logs one line for the model and one for an MTP draft, for example with Qwen3.8-Flash-Next-FP4-Local:

```
Abliteration on AbliteratedQwen4ExpForConditionalGeneration: lambda 1.5, ... residual writes in ... layers
Abliteration on AbliteratedQwen4ExpMTP: lambda 1.5, 5 residual writes in 1 layers
```

If the line is missing, the model is being served without abliteration.

### Architectures

| vLLM architecture | Models | Tested with |
|---|---|---|
| Qwen4Exp (`Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration`, `Qwen4ExpMTP`) | Qwen3.8-Flash-Next family | `nvidia/Qwen3.8-Flash-Next-FP4-Local` on vLLM 0.32.1rc1.dev467 (nightly), sm_80, tensor parallel, MTP on |

Loading does not depend on this list; GLM-5.3-Flash-FP4-Local, for example, loads but has no abliteration entry yet.

Each architecture is one entry in [`adapters.py`](https://github.com/lesj0610/fp4local-abliterated/blob/main/src/fp4local_abliterated/adapters.py): the vLLM module that defines it, the decoder layer class, and which submodules write into the residual stream. Everything else is shared. The plugin registers an `Abliterated<Arch>` subclass for every listed architecture through `ModelRegistry`, and attaches forward hooks after vLLM builds the model.

To add an architecture, list its residual writers (embedding, attention output, MLP or MoE output, and anything else added to the residual) as they are named in vLLM's model code, then check that the startup log counts one write per writer per layer.

### Where the direction for Qwen3.8-Flash-Next-FP4-Local came from

The direction used for [lesj0610/Qwen3.8-Flash-Next-FP4-Local-Abliterated](https://huggingface.co/lesj0610/Qwen3.8-Flash-Next-FP4-Local-Abliterated) was not captured by us. It is the direction of [windowsxp811203/Qwen3.8-Flash-Next-Abliterated](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated), a bf16 weight-edited build of the same base model, recovered exactly from the difference between its edited and original weights (the difference is rank-1). Their model card describes how it was captured: difference of mean residual activations on harmful and harmless prompts, after 37 of the 48 decoder layers. The method is from Arditi et al., [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (2024).

## Serving Qwen3.8-Flash-Next-FP4-Local

```bash
vllm serve <checkpoint> \
  --engram-config '{"cpu_offload": true}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

`--engram-config` is optional. It moves the PLE table from GPU memory into pinned host memory, which cannot be swapped out: plan for 36 GiB of free host RAM (the 27 GiB table, rounded up to powers of two by PyTorch's pinned-memory allocator).

With `--tensor-parallel-size` above 1, add `--mm-encoder-tp-mode data`. The vision MLP's intermediate size (4304) cannot be split across GPUs into NVFP4 blocks of 16, so each GPU keeps its own copy of the vision tower instead.

## Tests

```bash
pytest tests/
```

The CSF, shard lookup and CLI tests run on CPU. The projection tests need a CUDA GPU.

## License

Apache-2.0. `csf_scales.py` is adapted from vLLM (Apache-2.0).
