# ⚡ Flash Attention Implementation and Performance Study (Backward Pass)

This repository contains CUDA C++ implementations and a performance analysis for the backward pass of the Self-Attention mechanism, focusing on the highly optimized **Flash Attention** technique.

The work was developed as part of a **Topics in Machine Learning (TIML)** course at **National Taiwan University (NTU)**.

---

## 🚀 Project Overview

The primary goal of this project is to demonstrate and profile the drastic performance difference between **naive (textbook)**, **tiled (suboptimal)**, and **optimized Flash Attention** kernels for the Transformer self-attention backward pass.

The analysis is performed using a fixed head dimension D=64 and `half` (FP16) precision for all computations.

### Key Takeaways

1.  **Memory Bottleneck:** Naive implementations are severely limited by memory bandwidth due to reading the large N x N attention matrix up to D times from global memory.
2.  **L2 Thrashing:** For large sequence lengths N >> D, the naive kernel's working set size exceeds the GPU's L2 cache, leading to severe **cache thrashing**.
3.  **Flash Solution:** The optimized kernel eliminates this bottleneck by:
    * **Tiling:** Processing small blocks of Q, K, V and reusing them in fast **Shared Memory**.
    * **Recomputation:** Avoiding the storage and reading of the massive P matrix entirely by recomputing the necessary values on-the-fly.

---

## 📁 Repository Structure

| File Name | Description | Purpose |
| :--- | :--- | :--- |
| `flash_cuda_slow.cu` | **Performance Comparison Kernels** | Contains the three kernels profiled for comparison: 1. `naive_backward_kernel`, 2. `tiled_uncoalesced_kernel`, and 3. `flash_optimized_half`. This file provides the core speed contrast. |
| `flash_cuda_flash.cu` | **Block Size Optimization Study** | Focuses solely on the optimized `flash_optimized_half` kernel and benchmarks its performance across various tiling configurations (4x4, 8x8, 16x16, 32x32) to find the optimal block size for the target GPU. |
| `check_specs.cu` | **Hardware Analysis** | A utility program that reads and prints key GPU specifications (Compute Capability, L2 Cache Size, Shared Memory per Block) and calculates the **minimum working set size** for the attention problem to justify the memory-bound nature of the naive approach. |
| `README.md` | *This file.* | Project documentation and usage guide. |

---

## ⚙️ Dependencies

* **CUDA Toolkit:** Must be installed (tested with CUDA 11.x+).
* **A modern NVIDIA GPU:** Required to run the CUDA kernels and utilize `__half` types.

## 🛠️ Build and Run Instructions

The files can be compiled using the NVIDIA CUDA compiler (`nvcc`).

### 1. Compile `check_specs.cu`

Run this first to understand your GPU's memory constraints.

```bash
nvcc check_specs.cu -o check_specs
./check_specs
```
This README was generated using Gemini.
