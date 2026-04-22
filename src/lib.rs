//! # cortex-tensor
//!
//! Pure-Rust tensor + transformer + MoE building blocks. Zero GPU / CUDA /
//! Julia / framework dependencies.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`tensor`] | `Tensor` type + core ops |
//! | [`transformer`] | Transformer building blocks (attention, block, model) |
//! | [`moe`] | Mixture-of-Experts router + GGUF checkpoint bridge |
//! | [`types`] | Shared types used by `moe` |
//! | [`error`] | `CortexError` unified error type |

pub mod tensor;
pub mod transformer;
pub mod moe;
pub mod types;
pub mod error;

pub use tensor::Tensor;
pub use error::{CortexError, HybridError, Result};
