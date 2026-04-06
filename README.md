# SpikeLMo

SNN/LLM Fusion Framework — a candle-free neuromorphic inference and training system that bridges spiking neural networks with large language models.

## Overview

SpikeLMo is an upgrade and consolidation of my existing Rust and Julia libraries for neuromorphic computing. It integrates the **Spikenaut ecosystem** of crates into a unified framework for training and deploying SNN/LLM fusion models.

**LLM Choice**: This framework is going to use **allenai/OLMoE-1B-7B-0125-Instruct** as the base language model for SNN/LLM fusion experiments.

## Architecture

The framework consists of:

- **Custom Tensor & Transformer** — hand-rolled tensor operations (`src/tensor/`) and transformer components (`src/transformer/`) as candle-core/candle-nn replacements
- **SNN/LLM Fusion Layer** — novel bridge (`src/fusion.rs`) connecting neuromod's spiking networks with transformer hidden states
- **E-prop Training** — eligibility-propagation trainer (`src/training.rs`) using Spikenaut reward, encoder, and telemetry crates
- **FPGA Export** — Q8.8 fixed-point deployment pipeline (`src/fpga_export.rs`) via spikenaut-fpga

## Spikenaut Ecosystem Integration

SpikeLMo integrates the following local crates:

| Crate | Purpose | Used For |
|---|---|---|
| `neuromod` | High-performance SNN library | LIF neurons, STDP, neuromodulators |
| `spikenaut-encoder` | Sensory encoding pipelines | Telemetry → 16-channel spike trains |
| `spikenaut-reward` | Homeostatic reward computation | Mining efficiency dopamine, EMA smoothing |
| `spikenaut-fpga` | Q8.8 FPGA deployment | `.mem` file generation for hardware |
| `spikenaut-telemetry` | Hardware telemetry snapshots | GPU/CPU/mining data aggregation |
| `myelin-accelerator` | CUDA spiking kernels (optional) | GPU-dispatched SNN simulation |

## Upgrade Path

SpikeLMo represents an ongoing upgrade and consolidation of my existing Rust and Julia libraries for neuromorphic computing. This is an active development effort:

- **Active Integration** — I am still working on my Spikenaut ecosystem libraries to ensure proper configuration and integration with SpikeLMo
- **Legacy Code Mining** — I am reviewing my old Rust and Julia codebases to identify "hidden gems" — reusable components that can be extracted into standalone libraries
- **Consolidation Goal** — Merge functionality scattered across multiple repositories into a single orchestrator framework
- **Rust-Native Transition** — Replace Julia components with Rust implementations for better ecosystem integration
- **Candle-Free Design** — Custom tensor/transformer implementation avoiding external ML framework dependencies
- **Edition 2024** — Upgrading all crates to Rust edition 2024 for modern language features

## License

GPL-3.0
