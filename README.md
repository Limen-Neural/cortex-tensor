# cortex-tensor

Pure-Rust tensor, transformer, and Mixture-of-Experts building blocks. No CUDA, no Julia FFI, no framework dependencies — just `Vec<f32>` and honest math.

[![Rust](https://img.shields.io/badge/rust-edition%202024-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue)](./LICENSE-APACHE-2.0)

## Overview

`cortex-tensor` is a minimal, framework-free foundation for building transformer-based language models and MoE routers in Rust. It was surgically extracted from a larger hybrid codebase (`corinth-canal`) and then stripped of every GPU / CUDA / Julia / SNN-specific concern so it can stand alone as a reusable, open-source numerical kernel.

Design goals:

- **Zero GPU coupling.** No `cust`, no `libc` pinned-host registration, no `#[cfg(feature = "gpu")]` branches.
- **Zero framework dependency.** No `candle`, no `tch`, no `ort`. The tensor type is a row-major `Vec<f32>` with explicit shape + strides.
- **Small, auditable dependency set.** `serde`, `serde_json`, `thiserror`, `rand`, `rayon`, `memmap2`, `half` — nothing else.
- **Inference-ready MoE.** A GGUF checkpoint bridge with family-aware adapter resolution for OLMoE, Qwen3-MoE, Gemma-4, DeepSeek-2, and Llama-MoE.

## Architecture

```
src/
├── lib.rs            # re-exports Tensor, CortexError, HybridError, Result
├── error.rs          # CortexError + HybridError alias
├── types.rs          # EMBEDDING_DIM, ModelFamily, RoutingMode
├── tensor/
│   ├── mod.rs        # row-major Tensor { data, shape, strides }
│   └── ops.rs        # matmul, batched_matmul, causal_mask, softmax, ...
├── transformer/
│   ├── attention.rs  # MultiHeadAttention (scaled dot-product, causal mask)
│   ├── block.rs      # TransformerBlock (attn + MLP + LayerNorm)
│   ├── model.rs      # TransformerConfig + TransformerLM (decoder-only)
│   └── mod.rs
└── moe/
    ├── mod.rs        # OlmoeRouter public API, RoutingMode
    ├── adapter.rs    # model-family detection + tensor selection
    ├── checkpoint.rs # GGUF parser, mmap'd F32/F16/Q8_0/Q5_K access
    └── routing.rs    # softmax, top-k, L2 normalize, embedding resample
```

## Modules

### `tensor`

| Item | Purpose |
|---|---|
| `Tensor` | Row-major `f32` tensor (`data: Vec<f32>`, `shape`, `strides`), `Serialize`/`Deserialize`. |
| `ops::matmul` / `batched_matmul` | Cache-friendly tiled CPU matmul. |
| `ops::causal_mask` | Additive mask for auto-regressive attention. |
| `ops::softmax` / `layer_norm` | Standard building blocks. |

### `transformer`

| Item | Purpose |
|---|---|
| `MultiHeadAttention` | Multi-head scaled dot-product attention with causal masking. Weights are dense `Tensor`s; no external framework needed. |
| `TransformerBlock` | Attention → residual → MLP → residual, with pre-LayerNorm. |
| `TransformerLM` / `TransformerConfig` | Decoder-only transformer LM: token + positional embedding → N × block → LayerNorm → LM head. |

### `moe`

| Item | Purpose |
|---|---|
| `OlmoeRouter` | Family-aware MoE router. Loads a GGUF checkpoint, detects model family, and produces top-k expert selections. |
| `RoutingMode` | `StubUniform`, `DenseSim`, `SpikingSim` (simulation-only; no GPU dispatch). |
| `ModelFamily` | `Olmoe`, `Qwen3Moe`, `Gemma4`, `DeepSeek2`, `LlamaMoe`. |

Supported GGUF tensor types: `F32`, `F16`, `Q8_0`, `Q5_K`. `IQ3_S` is detected and rejected (for token embeddings) with a clear error so callers can fall back to `llama.cpp` prompt embeddings. For the preferred GPU synapse tensor (e.g. attn_q on qwen3_moe_iq3_m), unsupported quants now correctly route to a checkpoint-backed `routing-f32` source (using the F32 routing tensor) instead of synthetic fallback. See `synapse_source()`, `real_gpu_synapse_tensor_name()`, and `OlmoeRouter` metadata.

**Future formats (planning, see #9):** Safetensors support will arrive via a dedicated reusable `safetensors-parser` crate (header inspection + deterministic manifest + MoE candidate discovery), extracted as a one-way copy of reference logic from rmems/corinth-canal (see corinth-canal#116, engram-parser#10, and cortex #7/#8 for the GGUF precedent with engram-parser). No implementation or dependency is present yet — this keeps the reusable parser boundary clean. Cross-links and notes are maintained for alignment.

### GGUF adapter + synapse source + SAAQ flow (code paths)

- `OlmoeRouter::load` / `load_with_family_and_mode` → `probe_and_map` calls `resolve_adapter` (adapter.rs).
- `resolve_adapter` infers family from arch, validates routing tensor (must be F32 rank-2), selects token_embd or tok_embeddings, sets `preferred_gpu_synapse_tensor` to `blk.0.attn_q.weight` when present.
- Synapse source selection (updated for qwen3 IQ3_S): if attn_q is F16 rank-2 containing hidden_size (relaxed from strict square to support GQA) → `real`; elif attn_q present → `routing-f32` (real name = routing tensor name); else `synthetic-fallback`.
- Routing always uses `routing_tensor` via `checkpoint_gate_scores` (routing.rs) when checkpoint loaded (never synthetic for real loads).
- `extract_named_token_embedding_from_checkpoint` (checkpoint.rs) supports dequant for Q8_0/Q5_K (and F32/F16); IQ3_S errors for embeddings.
- Public metadata exposes `preferred_gpu_synapse_tensor_name`, `real_gpu_synapse_tensor_name`, `synapse_source` for SAAQ experiment / Surrogate_Viz consumers to choose dequant vs. synthetic path and load the right tensor (f16 path or f32 routing path).
- SAAQ artifacts (external): calibration runs consume the router to emit artifacts for viz; see labels on related issues for campaign.

## Install

```toml
[dependencies]
cortex-tensor = { git = "https://github.com/Limen-Neural/cortex-tensor", branch = "main" }
```

## Quick start

```rust
use cortex_tensor::tensor::ops::matmul;
use cortex_tensor::Tensor;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
let c = matmul(&a, &b);
assert_eq!(c.shape(), &[2, 2]);
```

Building a transformer block:

```rust
use cortex_tensor::transformer::{MultiHeadAttention, TransformerBlock};

let attn = MultiHeadAttention::new(/* dim */ 512, /* num_heads */ 8);
let block = TransformerBlock::new(/* dim */ 512, /* num_heads */ 8, /* mlp_dim */ 2048);
```

Loading an OLMoE-family GGUF and running the router:

```rust
use cortex_tensor::moe::{OlmoeRouter, RoutingMode};

let mut router = OlmoeRouter::load(
    "path/to/olmoe.gguf",
    RoutingMode::DenseSim,
    /* top_k */ 2,
)?;
let (experts, weights) = router.route_for_token(/* token_id */ 42)?;
```

## Optional Sentry monitoring

Enable with the `sentry` feature (uses sentry-rust 0.48):

```toml
[dependencies]
cortex-tensor = { git = "...", features = ["sentry"] }
```

Init guard example (call early in main or lib init; keep guard alive for duration of process).
Note: the consuming crate/binary must explicitly enable the `sentry` feature on its `cortex-tensor` dependency
(transitive features do not auto-activate). Then use the re-exported path (or add `sentry` as direct dep):

```rust
#[cfg(feature = "sentry")]
let _sentry_guard = cortex_tensor::sentry::init((
    "https://<key>@sentry.io/<project>",
    cortex_tensor::sentry::ClientOptions {
        release: cortex_tensor::sentry::release_name!(),
        environment: Some("production".into()),
        ..Default::default()
    },
));

// Your app code; errors auto captured when panics or cortex_tensor::sentry::capture_message etc used.
// (The re-export brings the full sentry crate API under the feature gate.)
```

When the feature is off the re-export is not present (guarded).

## Non-goals

- No GPU backend. Ever. If you need CUDA, consume this crate's `Tensor` into your own kernels.
- No automatic differentiation. This is an inference and forward-pass library.
- No tokenizer. Pair it with `tokenizers` or `llama.cpp`'s tokenizer of choice.
- No SNN / neuromorphic logic. Those live in upstream projects.

## Status

Extracted and compiling cleanly under Rust edition 2024. Public API is subject to change until `0.1.0` is tagged. Tests and benchmarks are incoming.

## License

This project is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

at your option.
