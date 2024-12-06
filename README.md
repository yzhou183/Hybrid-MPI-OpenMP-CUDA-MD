# Hybrid-MPI-OpenMP-CUDA-MD
# Hybrid MPI + OpenMP + CUDA MD Program

This project focuses on enhancing a Molecular Dynamics (MD) simulation program by leveraging a hybrid parallel computing model that integrates **MPI**, **OpenMP**, and **CUDA**. The goal is to improve the performance of the force calculation process and overall simulation efficiency for large-scale systems.

---

## Features

- **Hybrid Parallelism**: Combines distributed (MPI), shared memory (OpenMP), and GPU acceleration (CUDA) for efficient simulation.
- **CUDA Acceleration**: Offloads computationally intensive force calculations to GPUs, leveraging massive parallelism.
- **Non-blocking MPI Communication**: Optimizes data exchange between processes to overlap communication and computation.
- **Performance Profiling**: Analyzes bottlenecks using tools like `nvprof` and `Nsight Compute`.

---

## Motivation

Molecular Dynamics simulations are computationally intensive, especially for large systems where the complexity of force calculations grows quadratically with the number of particles. By utilizing hybrid parallel computing and GPU acceleration, this project aims to significantly reduce computation time while maintaining accuracy, scalability, and portability.

---

## Getting Started

### Prerequisites

- **Programming Languages**: C/C++ with CUDA
- **Libraries**:
  - MPI (e.g., OpenMPI or MPICH)
  - OpenMP
  - CUDA Toolkit
- **Environment**:
  - Linux or Unix-like system
  - NVIDIA GPU with CUDA support


