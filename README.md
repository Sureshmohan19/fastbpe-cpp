# fastbpe-cpp

A **high-performance, byte-level BPE (Byte Pair Encoding) tokenizer** written in C++ with a focus on **speed, cache locality, and minimal abstraction overhead**.

This project implements the *core BPE algorithm* from first principles and is optimized for fast training on large corpora. In benchmarks, it outperforms HuggingFace’s Rust tokenizer for the same workload.


## Why this exists

Most modern tokenizers prioritize:
- Unicode correctness
- Regex-based pretokenization
- Python / JSON integration
- Generality across many languages

That is useful — but expensive.

**fastbpe-cpp** focuses on a narrower goal:

> *Train BPE tokenizers as fast as possible at the byte level, using efficient data structures and predictable memory access.*

This makes it ideal as:
- a research baseline
- a systems reference implementation
- a core engine to build higher-level tooling on top of (Python bindings, streaming I/O, etc.)


## Features

- Fast byte-level BPE training
- Cache-friendly data structures
- Deterministic training and encoding
- Comprehensive test suite
- Compact binary model format
- No regex, no JSON, no Python dependency in the core


## What it does **not** do (by design)

This is intentionally **not** a full replacement for production tokenizers like `tokenizers`, `tiktoken`, or `sentencepiece`.

Current limitations:

- No Unicode normalization
- ASCII-only lexer (byte-level)
- No regex pretokenization
- No streaming I/O (loads full file into memory)
- No Python bindings (yet)

These are **conscious trade-offs**, not oversights.


## Architecture overview

At a high level:

1. **Lexing**
   - Input text is split into segments (whitespace / letters / digits / punctuation)
   - Each segment becomes a linked list of byte tokens

2. **Statistics collection**
   - Adjacent token pairs are counted using:
     - a linear-probing hash table
     - an intrusive inverted index stored in a single contiguous memory pool

3. **BPE training loop**
   - Always merges the most frequent valid pair
   - Updates only affected neighbors
   - Uses lazy staleness checks to avoid expensive recomputation

4. **Inference**
   - Builds a fast lookup table mapping `(token_a, token_b) → merge_rank`
   - Applies merges greedily (tiktoken-style)

The implementation is designed to be:
- branch-predictable
- allocation-light
- easy to reason about and debug


## Benchmark results

**Training on `tinyshakespeare.txt` with 5000 merges**

| Implementation | Mean time |
|---------------|-----------|
| fastbpe-cpp   | ~110 ms   |
| HuggingFace Rust tokenizer | ~233 ms |

**~2.1× faster** on this workload.

### Benchmark command

```bash
hyperfine \
  --warmup 3 \
  --runs 10 \
  './bin/fastbpe train data/tinyshakespeare.txt model.bin 5000 1' \
  'python eval/bench.py rust data/tinyshakespeare.txt 5000'
```

**Note:**

The HuggingFace tokenizer includes Unicode handling, regex-based pretokenization, and Python ↔ Rust boundary overhead.  

This comparison is **fair within the stated scope** of byte-level BPE training.

## Usage

### Build

```bash
g++ -std=c++17 -O3 -march=native -flto src/bpe.cpp -o bin/fastbpe
```

### Train

```bash
./bin/fastbpe train data/tinyshakespeare.txt model.bin 5000 1
```

### Encode

```bash
./bin/fastbpe encode model.bin "To be, or not to be"
```

### Decode

```bash
./bin/fastbpe decode model.bin 123 456 789
```

## Model format

Binary, little-endian, fixed layout:

*[u32 magic][u32 version]*

*[u32 vocab_size][u32 merge_count]*

*[MergeRule × merge_count]*

*[[u32 token_len][token_bytes] × vocab_size]*


## Tests

A comprehensive shell-based test suite is provided:

```bash
./eval/test_bpe.sh
```

### Test coverage list

- Round-trip encode / decode
- Determinism
- Empty input
- Whitespace & punctuation handling
- ASCII exhaustiveness
- Large input handling
- Corrupted model detection

## Contributing

Contributions are **very welcome**, especially in these areas:

### High-priority

- Python bindings (pybind11 or C API)
- Streaming I/O for large corpora
- Better byte-level lexer (FSM-based)
- Memory profiling & RSS reduction
- Linux support (only tested with macOS)

### Medium-priority

- Unicode-aware pretokenization
- Compatibility modes (GPT-2 / tiktoken)
- Benchmark extensions (encode throughput, memory)

### Low-priority

- CLI polish
- Windows support
- Documentation & diagrams

### Contribution guidelines

- Keep the core fast and simple
- Avoid unnecessary abstraction
- Prefer explicit data structures over generic containers
- Benchmark before and after changes

If you’re unsure where to start, open an issue — happy to discuss design trade-offs.

## Philosophy

This project values:

- **Clarity over cleverness**
- **Data locality over abstraction**
- **Measured performance over assumptions**

It is meant to be read, studied, extended, and — if needed — embedded inside larger systems.

## License

MIT License.

## Acknowledgements

Inspired by:

- Byte Pair Encoding (Gage, 1994)
- GPT-2 / tiktoken tokenizer behavior
- Karpathy’s *minbpe* (conceptual clarity)

If this repo is useful to you, a ⭐ star or issue is always appreciated 🙂