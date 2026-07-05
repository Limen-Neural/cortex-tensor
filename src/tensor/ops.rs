use super::Tensor;

/// Matrix multiply: [M×K] × [K×N] → [M×N]
///
/// Uses a cache-friendly tiled loop. When the `gpu` feature is enabled,
/// this will dispatch to a CUDA kernel instead.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "matmul: a must be 2-D");
    assert_eq!(b.ndim(), 2, "matmul: b must be 2-D");
    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    assert_eq!(k1, k2, "matmul: inner dims mismatch {k1} vs {k2}");

    let ad = a.data();
    let bd = b.data();
    let mut out = vec![0.0f32; m * n];

    // Tiled matmul for better cache behaviour
    const TILE: usize = 32;
    for i0 in (0..m).step_by(TILE) {
        for j0 in (0..n).step_by(TILE) {
            for p0 in (0..k1).step_by(TILE) {
                let i_end = (i0 + TILE).min(m);
                let j_end = (j0 + TILE).min(n);
                let p_end = (p0 + TILE).min(k1);
                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_ip = ad[i * k1 + p];
                        for j in j0..j_end {
                            out[i * n + j] += a_ip * bd[p * n + j];
                        }
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, &[m, n])
}

/// Batched matmul: [B×M×K] × [B×K×N] → [B×M×N]
/// If `b` is 2-D, broadcasts across batches.
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    if a.ndim() == 2 && b.ndim() == 2 {
        return matmul(a, b);
    }
    assert_eq!(a.ndim(), 3, "batched_matmul: a must be 2-D or 3-D");

    let batch = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];

    let b_is_batched = b.ndim() == 3;
    let n = if b_is_batched {
        assert_eq!(b.shape()[0], batch, "batched_matmul: batch mismatch");
        assert_eq!(b.shape()[1], k);
        b.shape()[2]
    } else {
        assert_eq!(b.ndim(), 2);
        assert_eq!(b.shape()[0], k);
        b.shape()[1]
    };

    let mut out = vec![0.0f32; batch * m * n];
    let ad = a.data();
    let bd = b.data();

    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = if b_is_batched { bi * k * n } else { 0 };
        let o_off = bi * m * n;
        for i in 0..m {
            for p in 0..k {
                let a_ip = ad[a_off + i * k + p];
                for j in 0..n {
                    out[o_off + i * n + j] += a_ip * bd[b_off + p * n + j];
                }
            }
        }
    }
    Tensor::from_vec(out, &[batch, m, n])
}

/// Layer normalization over the last axis.
/// Returns (normalized, mean, rstd) for backward pass if needed.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.ndim(), 2, "layer_norm: expects 2-D [batch, dim]");
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    assert_eq!(weight.numel(), cols);
    assert_eq!(bias.numel(), cols);

    let xd = x.data();
    let wd = weight.data();
    let bd = bias.data();
    let mut out = vec![0.0f32; rows * cols];

    for r in 0..rows {
        let off = r * cols;
        let row = &xd[off..off + cols];

        let mean: f32 = row.iter().sum::<f32>() / cols as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let rstd = 1.0 / (var + eps).sqrt();

        for c in 0..cols {
            out[off + c] = (row[c] - mean) * rstd * wd[c] + bd[c];
        }
    }
    Tensor::from_vec(out, &[rows, cols])
}

/// RMS normalization (used by LLaMA-family models).
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.ndim(), 2);
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    assert_eq!(weight.numel(), cols);

    let xd = x.data();
    let wd = weight.data();
    let mut out = vec![0.0f32; rows * cols];

    for r in 0..rows {
        let off = r * cols;
        let row = &xd[off..off + cols];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let rstd = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            out[off + c] = row[c] * rstd * wd[c];
        }
    }
    Tensor::from_vec(out, &[rows, cols])
}

/// Embedding lookup: [vocab_size, dim] indexed by token ids → [seq_len, dim]
pub fn embedding(table: &Tensor, ids: &[u32]) -> Tensor {
    assert_eq!(table.ndim(), 2);
    let dim = table.shape()[1];
    let td = table.data();
    let seq_len = ids.len();
    let mut out = vec![0.0f32; seq_len * dim];
    for (i, &id) in ids.iter().enumerate() {
        let src_off = id as usize * dim;
        out[i * dim..(i + 1) * dim].copy_from_slice(&td[src_off..src_off + dim]);
    }
    Tensor::from_vec(out, &[seq_len, dim])
}

/// Causal attention mask: upper triangle = -inf, lower triangle + diagonal = 0.
pub fn causal_mask(seq_len: usize) -> Tensor {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, &[seq_len, seq_len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x2() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_layer_norm() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        let y = layer_norm(&x, &w, &b, 1e-5);
        // Each row should have mean ≈ 0
        let row0_mean: f32 = y.data()[0..3].iter().sum::<f32>() / 3.0;
        assert!(row0_mean.abs() < 1e-4);
    }

    #[test]
    fn test_causal_mask() {
        let m = causal_mask(3);
        assert_eq!(m.data()[0], 0.0); // [0,0]
        assert!(m.data()[1].is_infinite()); // [0,1] = -inf
        assert_eq!(m.data()[4], 0.0); // [1,1]
    }

    #[test]
    fn test_embedding() {
        let table = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]);
        let ids = vec![2u32, 0u32];
        let out = embedding(&table, &ids);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data(), &[0.5, 0.6, 0.1, 0.2]);
    }
}
