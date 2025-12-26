# fastbpe-cpp

A high-performance, byte-level BPE tokenizer written in C++.

## Features
- Byte-level BPE training
- Cache-friendly data structures
- Faster than HuggingFace Rust tokenizer for this workload
- Deterministic and fully tested

## Status
Core engine is stable.
Python bindings, streaming I/O, and Unicode-aware lexing are welcome contributions.

## Benchmarks
See `eval/` for benchmark scripts.

## License
MIT
