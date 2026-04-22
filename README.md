# cortex-tensor

Pure-Rust tensor, transformer, and Mixture-of-Experts building blocks. No CUDA, no Julia FFI, no framework dependencies — just `Vec<f32>` and honest math.

[![Rust](https://img.shields.io/badge/rust-edition%202024-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](./LICENSE)

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

Supported GGUF tensor types: `F32`, `F16`, `Q8_0`, `Q5_K`. `IQ3_S` is detected and rejected with a clear error so callers can fall back to `llama.cpp` prompt embeddings.

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

## Non-goals

- No GPU backend. Ever. If you need CUDA, consume this crate's `Tensor` into your own kernels.
- No automatic differentiation. This is an inference and forward-pass library.
- No tokenizer. Pair it with `tokenizers` or `llama.cpp`'s tokenizer of choice.
- No SNN / neuromorphic logic. Those live in upstream projects.

## Status

Extracted and compiling cleanly under Rust edition 2024. Public API is subject to change until `0.1.0` is tagged. Tests and benchmarks are incoming.

## License

GPL-3.0 — see [LICENSE](./LICENSE).
