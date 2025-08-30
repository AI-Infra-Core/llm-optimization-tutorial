# LLM Optimization Tutorial

A beginner-friendly, hands-on tutorial for learning Large Language Model (LLM) training optimization techniques.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About The Project

This project provides a minimal, easy-to-understand codebase for fine-tuning Large Language Models. Our core philosophy is to explain complex optimization techniques with the simplest possible code.

To achieve this, **we use only foundational PyTorch and avoid high-level abstractions like Hugging Face `Trainer` or PyTorch Lightning**. This approach allows us to see exactly how optimization strategies are implemented, line by line.

We will start with the [Qwen3](https://huggingface.co/Qwen) series as our example model and build everything from the ground up.


## Features

* **Native PyTorch Focus**: Learn optimization mechanics directly in PyTorch.
* **Maximum Simplicity**: Every chapters is written to be as clear and readable as possible, focusing only on the essential logic.
* **From-Scratch Implementation**: We build the core training components ourselves, giving you a deeper understanding of the entire pipeline.
* **Hands-On & Extensible**: Provides runnable code that you can easily modify and extend for your own experiments.

## Tutorial Outline

* [Chapter 01: Training QWen3 0.6B Dense on a Single A100 GPU](./chapter-01/README.md)
    * **Pure PyTorch Model Implementation**: The `modeling_qwen3.py` is built using only foundational PyTorch layers (nn.Module, nn.Linear, etc.). It has zero dependencies on the Hugging Face transformers library.
    * **Padding-Free Training with Packed Sequences**: We completely eliminate wasted computation on padding tokens. The custom `packed_sequence_collate_fn` process variable-length sequences into a contiguous buffer.
    * **Batch Dimension Elimination**: The input tensors to the model are 2D `(total_tokens, hidden_size)` or 1D `(total_tokens,)`, not the traditional 3D `(batch_size, seq_len, hidden_size)`. This "unbatched" or "packed" approach is a modern technique used in high-performance libraries like FlashAttention to simplify kernel implementations.
* Chapter 02: Profiling and Performance Analysis. Coming soon.

## Contributing

Contributions are welcome! If you find a bug, have an idea, or want to add an explanation, please feel free to open an issue or submit a pull request.

