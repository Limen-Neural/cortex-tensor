pub mod ops;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Row-major dense tensor — candle-core replacement.
///
/// Stores `f32` data in a contiguous `Vec<f32>` with an arbitrary shape.
/// All operations are CPU-first; the `gpu` feature flag will add CUDA kernels later.
#[derive(Clone, Serialize, Deserialize)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    // ── Constructors ─────────────────────────────────────────────────

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "data len {} != product of shape {:?} ({})",
            data.len(),
            shape,
            numel
        );
        let strides = Self::compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self::from_vec(vec![0.0; numel], shape)
    }

    pub fn ones(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self::from_vec(vec![1.0; numel], shape)
    }

    pub fn full(shape: &[usize], val: f32) -> Self {
        let numel: usize = shape.iter().product();
        Self::from_vec(vec![val; numel], shape)
    }

    pub fn randn(shape: &[usize], mean: f32, std: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                // Box-Muller transform
                let u1: f32 = rng.gen_range(0.0f32..1.0).max(1e-7);
                let u2: f32 = rng.gen_range(0.0f32..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                mean + std * z
            })
            .collect();
        Self::from_vec(data, shape)
    }

    // ── Accessors ────────────────────────────────────────────────────

    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    // ── Reshape / view ───────────────────────────────────────────────

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let numel: usize = new_shape.iter().product();
        assert_eq!(numel, self.numel(), "reshape: element count mismatch");
        Self::from_vec(self.data.clone(), new_shape)
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose requires 2-D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Self::from_vec(out, &[cols, rows])
    }

    // ── Element-wise ops ─────────────────────────────────────────────

    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "add: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn sub(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "sub: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn mul(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "mul: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn scale(&self, s: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn add_scalar(&self, s: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x + s).collect();
        Self::from_vec(data, &self.shape)
    }

    // ── Activation functions ─────────────────────────────────────────

    pub fn relu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x.max(0.0)).collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn gelu(&self) -> Self {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                let c = (2.0f32 / std::f32::consts::PI).sqrt();
                x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            })
            .collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn silu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        Self::from_vec(data, &self.shape)
    }

    /// Fast sigmoid surrogate gradient (from soma-engine's E-prop)
    pub fn fast_sigmoid(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x / (1.0 + x.abs())).collect();
        Self::from_vec(data, &self.shape)
    }

    // ── Reductions ───────────────────────────────────────────────────

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.numel() as f32
    }

    pub fn max_val(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Softmax along the last axis (works on 1-D or flattened last dim of 2-D).
    pub fn softmax_last(&self) -> Self {
        assert!(self.ndim() <= 2, "softmax_last: max 2-D");
        if self.ndim() == 1 {
            let max_v = self.max_val();
            let exp: Vec<f32> = self.data.iter().map(|x| (x - max_v).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let data: Vec<f32> = exp.iter().map(|e| e / sum).collect();
            return Self::from_vec(data, &self.shape);
        }
        // 2-D: softmax each row
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let row_start = r * cols;
            let row = &self.data[row_start..row_start + cols];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|x| (x - max_v).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for c in 0..cols {
                data[row_start + c] = exp[c] / sum;
            }
        }
        Self::from_vec(data, &self.shape)
    }

    // ── Row / slice access ───────────────────────────────────────────

    pub fn row(&self, idx: usize) -> Self {
        assert_eq!(self.ndim(), 2);
        let cols = self.shape[1];
        let start = idx * cols;
        Self::from_vec(self.data[start..start + cols].to_vec(), &[cols])
    }

    // ── Internals ────────────────────────────────────────────────────

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        let n = self.data.len().min(8);
        for (i, v) in self.data[..n].iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.4}")?;
        }
        if self.data.len() > 8 {
            write!(f, ", ...")?;
        }
        write!(f, "])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec_and_shape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tt = t.transpose();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let s = t.softmax_last();
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]);
        let g = t.gelu();
        assert!((g.data()[0] - 0.0).abs() < 1e-4);
        assert!(g.data()[1] > 0.8); // GELU(1) ≈ 0.841
    }
}
