# Chapter 1: Training QWen3 0.6B Dense on a Single A100 GPU 

This chapter builds a complete pre-training pipeline from scratch. We will focus on understanding how the three core components work together.

## Directory Structure

```
.
├── configs/                # qwen3 model configuration files from huggingface
│   └── qwen3-0.6B.json
└── src/                    # Main source code
    ├── modeling_qwen3.py   # PyTorch implementation of the Qwen3 model architecture
    ├── data_loader.py      # A virtual dataset for dynamically generating token sequences
    ├── trainer.py          # The core Trainer class that encapsulates the training loop
    └── main.py             # The main entry point with the Click-based command-line interface
```

## Key Features

* **Pure PyTorch Model Implementation**: The `modeling_qwen3.py` is built using only foundational PyTorch layers (nn.Module, nn.Linear, etc.). It has zero dependencies on the Hugging Face transformers library.
* **Padding-Free Training with Packed Sequences**: We completely eliminate wasted computation on padding tokens. The custom `packed_sequence_collate_fn` process variable-length sequences into a contiguous buffer.
* **Batch Dimension Elimination**: The input tensors to the model are 2D `(total_tokens, hidden_size)` or 1D `(total_tokens,)`, not the traditional 3D `(batch_size, seq_len, hidden_size)`. This "unbatched" or "packed" approach is a modern technique used in high-performance libraries like FlashAttention to simplify kernel implementations.

## How to Run

Start Training: `python src/main.py configs/train_qwen3_0.6B_singleGPU.yaml`

## What's Next?

Profiling and Performance Analysis for this modle.