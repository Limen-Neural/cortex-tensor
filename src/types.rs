//! Minimal shared types used by the `moe` module.
//!
//! Ported from `corinth-canal` with SNN-specific items removed.

use serde::{Deserialize, Serialize};

/// Dimensionality of the dense embedding the projector hands to the router.
pub const EMBEDDING_DIM: usize = 2048;

/// Supported GGUF model families for the router bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelFamily {
    #[default]
    Olmoe,
    Qwen3Moe,
    Gemma4,
    DeepSeek2,
    LlamaMoe,
}

impl ModelFamily {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Olmoe => "olmoe",
            Self::Qwen3Moe => "qwen3_moe",
            Self::Gemma4 => "gemma4",
            Self::DeepSeek2 => "deepseek2",
            Self::LlamaMoe => "llama_moe",
        }
    }
}

/// Execution mode used by the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RoutingMode {
    StubUniform,
    DenseSim,
    #[default]
    SpikingSim,
}
