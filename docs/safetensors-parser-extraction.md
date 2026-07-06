# Safetensors Parser Extraction Design Note

**Related issues:**
- [cortex-tensor#9](https://github.com/Limen-Neural/cortex-tensor/issues/9) — coordination for reusable safetensors-parser
- [corinth-canal#116](https://github.com/rmems/corinth-canal/issues/116) — source extraction bootstrap
- [engram-parser#10](https://github.com/Limen-Neural/engram-parser/issues/10) — sibling extraction tracking
- Precedent: [engram-parser](https://github.com/Limen-Neural/engram-parser) for GGUF

## Background

`cortex-tensor` is currently GGUF-only (see `src/moe/checkpoint.rs`, `adapter.rs`, etc.).

Safetensors support exists as a reference implementation in the experimental `rmems/corinth-canal` repo. The reusable parts (header inspection, deterministic manifest generation, MoE router/expert candidate discovery) should be extracted to a dedicated zero/minimal-dep crate `safetensors-parser` (parallel to how engram-parser was extracted for GGUF).

**Key principle (from org rules):** One-way copy of code from inspiration. `corinth-canal` keeps an unmodified reference copy (per its `PROMOTION_RULES.md` and "frozen" status for some modules). No dependency from corinth-canal on the new crate. The new crate is for consumers like `cortex-tensor` (future multi-format MoE) and others.

This document provides a detailed spec of the extractable surface, based on direct inspection of corinth-canal.

## Extractable Public Surface (from corinth-canal)

Location in source: `src/moe/safetensors.rs` + `src/moe/safetensors/discovery.rs`

### Core Inspection Functions

- `pub fn inspect_safetensors_checkpoint(path: impl AsRef<Path>) -> Result<SafetensorsManifest>`
  - Accepts: single `.safetensors` file, `.safetensors.index.json`, or directory of shards.
  - Returns a deterministic, JSON-serializable manifest.
  - Handles sharded layouts, index validation, unreferenced shards, metadata.

- `pub fn write_safetensors_manifest(checkpoint_path: impl AsRef<Path>, output_path: impl AsRef<Path>) -> Result<SafetensorsManifest>`
  - Convenience wrapper that calls inspect + writes pretty JSON + conflict check.

### Key Types

```rust
pub struct SafetensorsManifest {
    pub manifest_version: u32,
    pub format: &'static str,
    pub checkpoint: SafetensorsCheckpointSource,
    pub tensors: Vec<SafetensorsTensorRecord>,
    pub candidates: SafetensorsCandidateSummary,
}

pub struct SafetensorsCheckpointSource {
    pub input_kind: String,           // "single_file", "directory", "index"
    pub index_file: Option<String>,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub metadata: BTreeMap<String, String>,
}

pub struct SafetensorsTensorRecord {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_size: usize,
    pub source_shard: String,
    pub data_offsets: [usize; 2],
    pub labels: Vec<&'static str>,    // e.g. from classify_tensor
}
```

### MoE Candidate Discovery (the reusable "intelligence")

```rust
pub struct SafetensorsCandidateSummary {
    pub detected_layout_family: Option<&'static str>,
    pub router_tensors: Vec<String>,
    pub expert_tensors: Vec<String>,
    pub router_candidates: Vec<SafetensorsRouterCandidate>,
    pub expert_groups: Vec<SafetensorsExpertGroup>,
}

pub struct SafetensorsRouterCandidate { ... }  // name, layer_hint, source_shard, shape, score, reasons
pub struct SafetensorsExpertGroup { ... }      // group_key, layer_hint, expert_indices, tensor_names, ...
```

- `pub(crate) fn discover_candidates(tensors: &[SafetensorsTensorRecord]) -> SafetensorsCandidateSummary`
- Internal helpers: `classify_tensor(name, &shape) -> Vec<&'static str>`, name parsing for `blk.N.ffn_*_exps`, scoring logic for router vs expert tensors, layout family detection (Olmoe, Qwen3Moe, Gemma4, etc.).

This logic is general and not tied to GGUF — perfect for extraction.

## Proposed Crate Structure (modeled on engram-parser)

```
safetensors-parser/
├── Cargo.toml          # zero/min deps: serde_json, memmap2 (for future mapped access?)
├── src/
│   ├── lib.rs
│   ├── error.rs
│   └── safetensors/
│       ├── mod.rs      # re-exports + inspect_safetensors_checkpoint, write_..., types
│       └── discovery.rs # candidate types + discover_candidates + classify
└── README.md
```

Public API surface (similar to engram):

- `inspect_safetensors_checkpoint`
- `write_safetensors_manifest`
- `SafetensorsManifest`, `SafetensorsTensorRecord`, `SafetensorsCandidateSummary`, `SafetensorsRouterCandidate`, `SafetensorsExpertGroup`
- Error type
- (Future) `MappedSafetensorsCheckpoint` or tensor accessors if needed for loading (but loading can stay in consumer like cortex for now)

## Integration Path for cortex-tensor

1. Add dependency on `safetensors-parser` (when published).
2. Extend `CheckpointBackend` enum and adapter resolution (see current `src/moe/mod.rs` and `adapter.rs`).
3. Use `inspect_*` for probing/manifests (analogous to GGUF).
4. Reuse candidate discovery for router/expert selection.
5. Keep `corinth-canal` as the reference implementation and test bed.

See also:
- Current GGUF handling in `src/moe/`
- `docs/` (when created) for coordination
- engram-parser's `src/moe/` for the GGUF parallel

## Non-Goals (for initial extraction)

- Do not move full `Mapped*` loading or GPU registration into the parser crate initially.
- Do not change public API of cortex-tensor in this coordination phase.
- No dependency from corinth-canal.

## Next Steps (post this PR)

- Create `Limen-Neural/safetensors-parser` repo (or under engram? but separate per precedent).
- Port/adapt the inspect + discovery code (one-way).
- Publish + wire into cortex-tensor (see #7 traits for reusability).
- Update corinth-canal to optionally depend (or keep reference copy only).

## References

- corinth-canal `src/moe/safetensors.rs` and `safetensors/discovery.rs`
- corinth-canal `examples/safetensors_manifest.rs`
- corinth-canal `docs/ARCHITECTURE.md`, `MODEL_SOURCE_VERIFICATION_CHECKLIST.md`
- engram-parser README (ecosystem section)
- cortex-tensor#9, #7, #14

*This note serves as the detailed extraction spec for issue #9.*
