# CudaRF — From-Scratch CUDA Reconstruction

This repository contains a from-scratch CUDA reconstruction of CudaRF,
based on the algorithmic and GPU design described by Grahn, Lavesson,
Lapajne, and Slat in their 2010 and 2011 papers.

The implementation reconstructs the authors' tree-level GPU parallelism,
iterative stack-based tree construction, entropy-based splitting, bootstrap
bagging, OOB evaluation, and GPU prediction pipeline, with documented
adaptations required for modern CUDA toolchains (CUDA 12, sm_86).

This reconstruction is intended for reproducibility and comparative empirical
evaluation. It is not the original source code released by the original authors.

---

## References

- Grahn, Lavesson, Lapajne, Slat (2010). "A CUDA Implementation of Random
  Forests — Early Results." Third Swedish Workshop on Multi-core Computing.
- Grahn, Lavesson, Lapajne, Slat (2011). "CudaRF: A CUDA-based Implementation
  of Random Forests." AICCSA 2011.

---

## Build

```bash
nvcc -O2 -std=c++14 -arch=sm_86 cudarf.cu -o cudarf
```

Tested on WSL2 (Ubuntu), CUDA 12.4, NVIDIA RTX 3070 (sm_86).

---

## Run

```bash
./cudarf data.csv --trees 100 --k sqrt --depth 6 --seed 42 --trainfrac 0.8
./cudarf --help
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--trees N` | 100 | Number of trees |
| `--k sqrt\|log2\|all\|N` | sqrt | mtry: candidate features per split |
| `--depth N` | 6 | Maximum tree depth |
| `--seed N` | 42 | RNG seed for train/test split |
| `--trainfrac F` | 0.8 | Fraction of data used for training |
| `--min-samples N` | 2 | Minimum samples to split a node |
| `--header` | — | Force header row detection |
| `--no-header` | — | Suppress header row detection |

---

## What is reconstructed

The following design elements from the original papers are reproduced
faithfully:

- **Tree-level parallelism:** one CUDA thread builds one entire tree,
  exactly as described in both papers.
- **Iterative tree construction:** an explicit stack-based builder on the
  device, matching the authors' GT220 design (compute capability 1.2 did
  not support device-side recursion).
- **Splitting criterion:** entropy / information gain, the authors' explicit
  choice over Gini to shift cost from training to classification.
- **Bootstrap bagging:** one bagging kernel launch assigns each tree's
  bootstrap sample.
- **OOB classification and error estimation.**
- **GPU prediction kernel:** one thread per tree reads test data through a
  texture object (the CUDA-12-compatible equivalent of the papers' texture
  references).
- **GPU memory optimizations:** constant memory for configuration scalars,
  pinned host memory (`cudaHostAlloc`) for PCIe transfer, fast math
  intrinsics (`__logf`).
- **8-step host/device pipeline** from Fig. 2 of the 2011 paper, labelled
  in the source and terminal output.

---

## Documented deviations

| Aspect | Original paper | This reconstruction |
|--------|---------------|---------------------|
| Bagging/mtry RNG | CUDA SDK Mersenne Twister (legacy, no longer ships with CUDA 12) | Per-thread splitmix64-style generator |
| Texture memory API | Texture references (deprecated in CUDA 12) | Texture objects (same read-only cache path) |
| Train/test split | Not specified (evaluation used k-fold CV) | Shared splitmix64 stratified split (benchmarking methodology decision) |

---

## Output

Running `./cudarf` on a dataset prints GPU timing and accuracy across
seven sections:

- **7.1** Correctness and reconstruction validation
- **7.2** Training performance (H2D, bagging kernel, tree-build kernel, GPU Train, D2H, Total GPU Train)
- **7.3** Prediction performance (inference time, KSps throughput)
- **7.4** Tree structure analysis (nodes, leaves, depth, samples/leaf)
- **7.5** How tree structure affects GPU behaviour
- **7.6** Memory consumption and resource requirements

CPU/GPU performance comparison is performed externally by the benchmarking
methodology and is not part of the CudaRF executable. The executable reports
GPU measurements only.

---

## CPU/GPU comparison

CPU baseline comparison is handled outside this repository:

```
CudaRF GPU result  ─────┐
                        ├──> External benchmarking / survey reporting
CPU baseline result ────┘
```

Do not expect `cudarf` to launch or depend on any separate CPU baseline
executable.
