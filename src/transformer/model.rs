use super::block::TransformerBlock;
use crate::tensor::Tensor;
use crate::tensor::ops::{embedding, layer_norm, matmul};
use serde::{Deserialize, Serialize};

/// Configuration for a decoder-only transformer LM.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
}

impl TransformerConfig {
    /// A small config for testing (~2M params).
    pub fn tiny() -> Self {
        Self {
            vocab_size: 4096,
            dim: 128,
            num_heads: 4,
            num_layers: 4,
            ff_dim: 512,
            max_seq_len: 256,
        }
    }

    /// Approximate OLMo-1B scale config.
    pub fn olmo_1b() -> Self {
        Self {
            vocab_size: 50280,
            dim: 2048,
            num_heads: 16,
            num_layers: 16,
            ff_dim: 8192,
            max_seq_len: 2048,
        }
    }

    pub fn estimated_params(&self) -> usize {
        let embed = self.vocab_size * self.dim;
        let per_block = {
            let attn = 4 * self.dim * self.dim + 4 * self.dim;
            let ffn = 2 * self.dim * self.ff_dim + self.dim + self.ff_dim;
            let ln = 4 * self.dim;
            attn + ffn + ln
        };
        let final_ln = 2 * self.dim;
        let lm_head = self.dim * self.vocab_size;
        embed + self.num_layers * per_block + final_ln + lm_head
    }
}

/// Decoder-only transformer language model — candle-free.
///
/// Architecture: token embedding + positional embedding → N × TransformerBlock → LayerNorm → LM head
#[derive(Clone, Serialize, Deserialize)]
pub struct TransformerLM {
    pub config: TransformerConfig,
    pub tok_embed: Tensor, // [vocab_size, dim]
    pub pos_embed: Tensor, // [max_seq_len, dim]
    pub blocks: Vec<TransformerBlock>,
    pub final_ln_w: Tensor, // [dim]
    pub final_ln_b: Tensor, // [dim]
    pub lm_head: Tensor,    // [dim, vocab_size]
}

impl TransformerLM {
    pub fn new(cfg: TransformerConfig) -> Self {
        let scale = 0.02;
        let blocks: Vec<TransformerBlock> = (0..cfg.num_layers)
            .map(|_| TransformerBlock::new(cfg.dim, cfg.num_heads, cfg.ff_dim))
            .collect();

        Self {
            tok_embed: Tensor::randn(&[cfg.vocab_size, cfg.dim], 0.0, scale),
            pos_embed: Tensor::randn(&[cfg.max_seq_len, cfg.dim], 0.0, scale),
            blocks,
            final_ln_w: Tensor::ones(&[cfg.dim]),
            final_ln_b: Tensor::zeros(&[cfg.dim]),
            lm_head: Tensor::randn(&[cfg.dim, cfg.vocab_size], 0.0, scale),
            config: cfg,
        }
    }

    /// Forward pass: token_ids → logits [seq_len, vocab_size]
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        assert!(seq_len <= self.config.max_seq_len);

        // Token + positional embeddings
        let tok = embedding(&self.tok_embed, token_ids);
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos = embedding(&self.pos_embed, &pos_ids);
        let mut x = tok.add(&pos);

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // Final layer norm
        let x = layer_norm(&x, &self.final_ln_w, &self.final_ln_b, 1e-5);

        // LM head: [seq_len, dim] × [dim, vocab] → [seq_len, vocab]
        matmul(&x, &self.lm_head)
    }

    /// Get the hidden state after all transformer blocks (before LM head).
    /// Used by the SNN/LLM fusion layer.
    pub fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        assert!(seq_len <= self.config.max_seq_len);

        let tok = embedding(&self.tok_embed, token_ids);
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos = embedding(&self.pos_embed, &pos_ids);
        let mut x = tok.add(&pos);

        for block in &self.blocks {
            x = block.forward(&x);
        }

        layer_norm(&x, &self.final_ln_w, &self.final_ln_b, 1e-5)
    }

    pub fn param_count(&self) -> usize {
        self.config.estimated_params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_forward() {
        let cfg = TransformerConfig::tiny();
        let model = TransformerLM::new(cfg.clone());
        let ids = vec![1u32, 42, 100, 7];
        let logits = model.forward(&ids);
        assert_eq!(logits.shape(), &[4, cfg.vocab_size]);
    }

    #[test]
    fn test_hidden_states() {
        let cfg = TransformerConfig::tiny();
        let model = TransformerLM::new(cfg.clone());
        let ids = vec![1u32, 2, 3];
        let h = model.hidden_states(&ids);
        assert_eq!(h.shape(), &[3, cfg.dim]);
    }
}
