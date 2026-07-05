// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF format constants.

pub(crate) const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
pub(crate) const GGUF_VERSION: u32 = 3;
pub(crate) const GGML_TYPE_F32: u32 = 0;
pub(crate) const GGML_TYPE_F16: u32 = 1;
pub(crate) const GGML_TYPE_Q8_0: u32 = 8;
pub(crate) const GGML_TYPE_Q5_K: u32 = 13;
pub(crate) const GGML_TYPE_IQ3_S: u32 = 21;
pub(crate) const GGUF_VALUE_TYPE_UINT8: u32 = 0;
pub(crate) const GGUF_VALUE_TYPE_INT8: u32 = 1;
pub(crate) const GGUF_VALUE_TYPE_UINT16: u32 = 2;
pub(crate) const GGUF_VALUE_TYPE_INT16: u32 = 3;
pub(crate) const GGUF_VALUE_TYPE_UINT32: u32 = 4;
pub(crate) const GGUF_VALUE_TYPE_INT32: u32 = 5;
pub(crate) const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
pub(crate) const GGUF_VALUE_TYPE_BOOL: u32 = 7;
pub(crate) const GGUF_VALUE_TYPE_STRING: u32 = 8;
pub(crate) const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
pub(crate) const GGUF_VALUE_TYPE_UINT64: u32 = 10;
pub(crate) const GGUF_VALUE_TYPE_INT64: u32 = 11;
pub(crate) const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;
