// SPDX-License-Identifier: MIT OR Apache-2.0

// NOTE: This file contains GGUF parsing logic. Some LOC metrics are high due to
// the complexity of supporting multiple quantization formats (F32, F16, Q8_0, Q5_K, etc.)
// and memory-mapped access. Refactoring is tracked separately.

//! GGUF checkpoint parsing and mapped tensor access for the router bridge.

use super::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ3_S, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGUF_MAGIC,
    GGUF_VALUE_TYPE_ARRAY, GGUF_VALUE_TYPE_BOOL, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64,
    GGUF_VALUE_TYPE_INT8, GGUF_VALUE_TYPE_INT16, GGUF_VALUE_TYPE_INT32, GGUF_VALUE_TYPE_INT64,
    GGUF_VALUE_TYPE_STRING, GGUF_VALUE_TYPE_UINT8, GGUF_VALUE_TYPE_UINT16, GGUF_VALUE_TYPE_UINT32,
    GGUF_VALUE_TYPE_UINT64, GGUF_VERSION,
};
use super::dequant;
use crate::error::{HybridError, Result};
use memmap2::{MmapMut, MmapOptions};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::slice;

#[derive(Debug)]
pub(super) struct MappedGgufCheckpoint {
    mmap: MmapMut,
    tensors: HashMap<String, GgufTensorInfo>,
    metadata: GgufMetadata,
}

#[derive(Debug, Clone)]
pub(super) struct GgufTensorInfo {
    pub(super) dims: Vec<usize>,
    pub(super) ggml_type: u32,
    pub(super) relative_offset: usize,
    pub(super) absolute_offset: usize,
    pub(super) n_elements: usize,
}

pub(super) struct ParsedCheckpointLayout {
    pub(super) metadata: GgufMetadata,
    pub(super) tensors: HashMap<String, GgufTensorInfo>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct GgufMetadata {
    architecture: String,
    quantization: String,
    #[allow(dead_code)]
    strings: HashMap<String, String>,
    numerics: HashMap<String, u64>,
}

struct GgufCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

pub(super) fn extract_named_token_embedding_from_checkpoint(
    checkpoint: &mut MappedGgufCheckpoint,
    tensor_name: &str,
    path: &str,
    token_id: usize,
) -> Result<Vec<f32>> {
    checkpoint.extract_token_embedding(tensor_name, path, token_id)
}

impl GgufMetadata {
    pub(super) fn architecture(&self) -> &str {
        if self.architecture.is_empty() {
            "unknown"
        } else {
            &self.architecture
        }
    }

    pub(super) fn quantization(&self) -> &str {
        &self.quantization
    }

    pub(super) fn numeric(&self, key: &str) -> Option<usize> {
        self.numerics.get(key).copied().map(|v| v as usize)
    }
}

impl MappedGgufCheckpoint {
    fn extract_token_embedding(
        &mut self,
        tensor_name: &str,
        path: &str,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let info = self.tensor_info(tensor_name, path)?.clone();
        let d0 = info.dims[0];
        let d1 = info.dims.get(1).copied().unwrap_or(0);

        match info.ggml_type {
            GGML_TYPE_F32 => {
                let weights = self.f32_tensor(tensor_name, path)?;
                if token_id >= d1 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d1,
                        got: token_id,
                    });
                }
                Ok(weights[token_id * d0..token_id * d0 + d0].to_vec())
            }
            GGML_TYPE_F16 => {
                let values = self.u16_tensor_values(&info, path, tensor_name)?;
                if token_id >= d1 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d1,
                        got: token_id,
                    });
                }
                Ok(values[token_id * d0..token_id * d0 + d0]
                    .iter()
                    .map(|&b| dequant::f16_to_f32(b))
                    .collect())
            }
            GGML_TYPE_Q8_0 => {
                dequant::dequantize_row_q8_0(self.row_bytes(&info, token_id, path, tensor_name)?, d0)
            }
            GGML_TYPE_Q5_K => {
                dequant::dequantize_row_q5_k(self.row_bytes(&info, token_id, path, tensor_name)?, d0)
            }
            GGML_TYPE_IQ3_S => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' uses IQ3_S token embeddings; use llama.cpp prompt embeddings for this checkpoint"
            ))),
            other => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' has unsupported ggml_type={other}"
            ))),
        }
    }
}

pub(super) fn probe_and_map_checkpoint(path: &str) -> Result<(GgufMetadata, MappedGgufCheckpoint)> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: e.to_string(),
        })?;
    // SAFETY: The file is a valid, readable file descriptor opened above.
    // `map_copy` creates a private copy-on-write mapping that does not
    // write back to the underlying file.  The writable mapping is required
    // by `cuMemHostRegister_v2`, which expects a non-const pointer even
    // though it does not modify the memory contents.
    let mmap =
        unsafe { MmapOptions::new().map_copy(&file) }.map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("copy-on-write mmap failed: {e}"),
        })?;

    let parsed = parse_checkpoint_layout(&mmap, path)?;

    Ok((
        parsed.metadata.clone(),
        MappedGgufCheckpoint {
            mmap,
            tensors: parsed.tensors,
            metadata: parsed.metadata,
        },
    ))
}

pub(super) fn parse_checkpoint_layout(bytes: &[u8], path: &str) -> Result<ParsedCheckpointLayout> {
    let mut cursor = GgufCursor::new(bytes);

    let magic = cursor.read_exact(4, path)?;
    if magic != GGUF_MAGIC {
        return Err(HybridError::UnsupportedFormat(format!(
            "unrecognised model magic bytes: {magic:?}"
        )));
    }

    let version = cursor.read_u32(path)?;
    if version != GGUF_VERSION {
        return Err(HybridError::UnsupportedFormat(format!(
            "unsupported GGUF version {version}; expected {GGUF_VERSION}"
        )));
    }

    // Sanity-bound the header counts to prevent OOM allocation from malformed files.
    const MAX_TENSOR_COUNT: usize = 100_000;
    const MAX_KV_COUNT: usize = 100_000;
    const MAX_TENSOR_DIMS: usize = 8;

    let tensor_count_raw = cursor.read_u64(path)?;
    if tensor_count_raw > MAX_TENSOR_COUNT as u64 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor_count {tensor_count_raw} exceeds maximum allowed {MAX_TENSOR_COUNT}"
        )));
    }
    let tensor_count = tensor_count_raw as usize;

    let kv_count_raw = cursor.read_u64(path)?;
    if kv_count_raw > MAX_KV_COUNT as u64 {
        return Err(HybridError::UnsupportedFormat(format!(
            "kv_count {kv_count_raw} exceeds maximum allowed {MAX_KV_COUNT}"
        )));
    }
    let kv_count = kv_count_raw as usize;

    let mut alignment = 32usize;
    let mut file_type = None;
    let mut strings = HashMap::new();
    let mut numerics = HashMap::new();

    for _ in 0..kv_count {
        let key = cursor.read_string(path)?;
        let value_type = cursor.read_u32(path)?;
        match key.as_str() {
            "general.alignment" => alignment = cursor.read_numeric_as_usize(value_type, path)?,
            "general.file_type" => {
                let value = cursor.read_numeric_as_u32(value_type, path)?;
                file_type = Some(value);
                numerics.insert(key, value as u64);
            }
            "general.architecture" => {
                let value = cursor.read_string(path)?;
                strings.insert(key, value);
            }
            _ => {
                if let Some(value) = cursor.read_numeric_value(value_type, path)? {
                    numerics.insert(key, value);
                } else if value_type == GGUF_VALUE_TYPE_STRING {
                    strings.insert(key, cursor.read_string(path)?);
                } else {
                    cursor.skip_value(value_type, path)?;
                }
            }
        }
    }

    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.read_string(path)?;
        let n_dims_raw = cursor.read_u32(path)? as usize;
        if n_dims_raw > MAX_TENSOR_DIMS {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has {n_dims_raw} dims, which exceeds maximum allowed {MAX_TENSOR_DIMS}"
            )));
        }
        let n_dims = n_dims_raw;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(cursor.read_u64(path)? as usize);
        }
        let ggml_type = cursor.read_u32(path)?;
        let relative_offset = cursor.read_u64(path)? as usize;
        let n_elements = dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' element count overflow"),
            })?;
        tensors.insert(
            name,
            GgufTensorInfo {
                dims,
                ggml_type,
                relative_offset,
                absolute_offset: 0,
                n_elements,
            },
        );
    }

    let tensor_data_offset = dequant::align_up(cursor.offset, alignment);
    for tensor in tensors.values_mut() {
        tensor.absolute_offset = tensor_data_offset + tensor.relative_offset;
    }

    Ok(ParsedCheckpointLayout {
        metadata: GgufMetadata {
            architecture: strings
                .get("general.architecture")
                .cloned()
                .unwrap_or_else(|| "unknown".into()),
            quantization: dequant::quantization_label(file_type),
            strings,
            numerics,
        },
        tensors,
    })
}

impl MappedGgufCheckpoint {
    pub(super) fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub(super) fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub(super) fn find_first_tensor_with_suffix(&self, suffix: &str) -> Option<&str> {
        let mut matches: Vec<&str> = self
            .tensors
            .keys()
            .map(String::as_str)
            .filter(|name| name.ends_with(suffix))
            .collect();
        matches.sort_unstable_by_key(|name| dequant::tensor_block_sort_key(name));
        matches.into_iter().next()
    }

    pub(super) fn tensor_info<'a>(&'a self, name: &str, path: &str) -> Result<&'a GgufTensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })
    }

    pub(super) fn f32_tensor<'a>(&'a self, name: &str, path: &str) -> Result<&'a [f32]> {
        let info = self.tensor_info(name, path)?;
        if info.ggml_type != GGML_TYPE_F32 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be F32, got ggml_type={}",
                info.ggml_type
            )));
        }

        let start = info.absolute_offset;
        let end = start + info.n_elements * std::mem::size_of::<f32>();
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' extends beyond mapped file"),
            });
        }

        // SAFETY: `start` is a valid byte offset into the mmap and `end` is
        // checked against `mmap.len()` above.  F32 alignment is guaranteed
        // because GGUF aligns all tensor data to at least 32 bytes (enforced
        // by the `alignment` field parsed from the file header).  The returned
        // slice borrows `self` for lifetime `'a`, keeping the mmap alive.
        let ptr = unsafe { self.mmap.as_ptr().add(start) as *const f32 };
        Ok(unsafe { slice::from_raw_parts(ptr, info.n_elements) })
    }

    fn u16_tensor_values(
        &self,
        info: &GgufTensorInfo,
        path: &str,
        tensor_name: &str,
    ) -> Result<Vec<u16>> {
        let byte_start = info.absolute_offset;
        let byte_end = byte_start + info.n_elements * 2;
        if byte_end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' extends beyond mapped file"),
            });
        }
        Ok(self.mmap[byte_start..byte_end]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    fn row_bytes<'a>(
        &'a self,
        info: &GgufTensorInfo,
        row_idx: usize,
        path: &str,
        tensor_name: &str,
    ) -> Result<&'a [u8]> {
        let n_rows = info.dims.get(1).copied().unwrap_or(0);
        if row_idx >= n_rows {
            return Err(HybridError::InputLengthMismatch {
                expected: n_rows,
                got: row_idx,
            });
        }

        let row_size = dequant::tensor_row_size(info.ggml_type, info.dims[0])?;
        let start =
            info.absolute_offset
                .checked_add(row_idx.checked_mul(row_size).ok_or_else(|| {
                    HybridError::ModelLoad {
                        path: path.to_owned(),
                        reason: format!("tensor '{tensor_name}' row offset overflow"),
                    }
                })?)
                .ok_or_else(|| HybridError::ModelLoad {
                    path: path.to_owned(),
                    reason: format!("tensor '{tensor_name}' row offset overflow"),
                })?;
        let end = start
            .checked_add(row_size)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' row offset overflow"),
            })?;
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' row extends beyond mapped file"),
            });
        }
        Ok(&self.mmap[start..end])
    }

    /// Pure-CPU F16 tensor access. Returns an owned `Vec<u16>` of the
    /// tensor's raw 16-bit values (no GPU pin-registration).
    #[allow(dead_code)]
    pub(super) fn f16_tensor_values(&self, name: &str, path: &str) -> Result<Vec<u16>> {
        let info = self.tensor_info(name, path)?.clone();
        if info.ggml_type != GGML_TYPE_F16 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be F16, got ggml_type={}",
                info.ggml_type
            )));
        }
        self.u16_tensor_values(&info, path, name)
    }
}


impl<'a> GgufCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, len: usize, path: &str) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "cursor overflow".into(),
            })?;
        if end > self.bytes.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "unexpected EOF while parsing GGUF".into(),
            });
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self, path: &str) -> Result<u8> {
        Ok(self.read_exact(1, path)?[0])
    }

    fn read_u16(&mut self, path: &str) -> Result<u16> {
        let bytes = self.read_exact(2, path)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self, path: &str) -> Result<u32> {
        let bytes = self.read_exact(4, path)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_u64(&mut self, path: &str) -> Result<u64> {
        let bytes = self.read_exact(8, path)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_i16(&mut self, path: &str) -> Result<i16> {
        Ok(self.read_u16(path)? as i16)
    }

    fn read_i32(&mut self, path: &str) -> Result<i32> {
        Ok(self.read_u32(path)? as i32)
    }

    fn read_i64(&mut self, path: &str) -> Result<i64> {
        Ok(self.read_u64(path)? as i64)
    }

    fn read_string(&mut self, path: &str) -> Result<String> {
        let len = self.read_u64(path)? as usize;
        let bytes = self.read_exact(len, path)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("invalid UTF-8 in GGUF string: {e}"),
        })
    }

    fn read_numeric_as_u32(&mut self, value_type: u32, path: &str) -> Result<u32> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 => Ok(self.read_u8(path)? as u32),
            GGUF_VALUE_TYPE_INT8 => Ok(self.read_u8(path)? as i8 as i32 as u32),
            GGUF_VALUE_TYPE_UINT16 => Ok(self.read_u16(path)? as u32),
            GGUF_VALUE_TYPE_INT16 => Ok(self.read_i16(path)? as i32 as u32),
            GGUF_VALUE_TYPE_UINT32 => self.read_u32(path),
            GGUF_VALUE_TYPE_INT32 => Ok(self.read_i32(path)? as u32),
            GGUF_VALUE_TYPE_UINT64 => Ok(self.read_u64(path)? as u32),
            GGUF_VALUE_TYPE_INT64 => Ok(self.read_i64(path)? as u32),
            _ => Err(HybridError::UnsupportedFormat(format!(
                "GGUF numeric conversion from type {value_type} is not supported"
            ))),
        }
    }

    fn read_numeric_as_usize(&mut self, value_type: u32, path: &str) -> Result<usize> {
        Ok(self.read_numeric_as_u32(value_type, path)? as usize)
    }

    fn read_numeric_value(&mut self, value_type: u32, path: &str) -> Result<Option<u64>> {
        let value = match value_type {
            GGUF_VALUE_TYPE_UINT8 => Some(self.read_u8(path)? as u64),
            GGUF_VALUE_TYPE_INT8 => Some(self.read_u8(path)? as i8 as i64 as u64),
            GGUF_VALUE_TYPE_UINT16 => Some(self.read_u16(path)? as u64),
            GGUF_VALUE_TYPE_INT16 => Some(self.read_i16(path)? as i64 as u64),
            GGUF_VALUE_TYPE_UINT32 => Some(self.read_u32(path)? as u64),
            GGUF_VALUE_TYPE_INT32 => Some(self.read_i32(path)? as i64 as u64),
            GGUF_VALUE_TYPE_UINT64 => Some(self.read_u64(path)?),
            GGUF_VALUE_TYPE_INT64 => Some(self.read_i64(path)? as u64),
            GGUF_VALUE_TYPE_BOOL => Some(self.read_u8(path)? as u64),
            GGUF_VALUE_TYPE_FLOAT32 => Some(self.read_u32(path)? as u64),
            GGUF_VALUE_TYPE_FLOAT64 => Some(self.read_u64(path)?),
            GGUF_VALUE_TYPE_STRING | GGUF_VALUE_TYPE_ARRAY => None,
            other => {
                return Err(HybridError::UnsupportedFormat(format!(
                    "unsupported GGUF value type {other}"
                )));
            }
        };
        Ok(value)
    }

    fn skip_value(&mut self, value_type: u32, path: &str) -> Result<()> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_BOOL => {
                self.read_exact(1, path)?;
            }
            GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 => {
                self.read_exact(2, path)?;
            }
            GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_FLOAT32 => {
                self.read_exact(4, path)?;
            }
            GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_FLOAT64 => {
                self.read_exact(8, path)?;
            }
            GGUF_VALUE_TYPE_STRING => {
                let _ = self.read_string(path)?;
            }
            GGUF_VALUE_TYPE_ARRAY => {
                let nested_type = self.read_u32(path)?;
                let len = self.read_u64(path)? as usize;
                for _ in 0..len {
                    self.skip_value(nested_type, path)?;
                }
            }
            _ => {
                return Err(HybridError::UnsupportedFormat(format!(
                    "unsupported GGUF value type {value_type}"
                )));
            }
        }
        Ok(())
    }
}
