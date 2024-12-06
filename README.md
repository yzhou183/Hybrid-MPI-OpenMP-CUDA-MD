# Hybrid MPI + OpenMP + CUDA MD Program

This project focuses on enhancing a Molecular Dynamics (MD) simulation program by leveraging a hybrid parallel computing model that integrates **MPI**, **OpenMP**, and **CUDA**. The goal is to improve the performance of the force calculation process and overall simulation efficiency for large-scale systems.

---

## Features

- **Hybrid Parallelism**: Combines distributed (MPI), shared memory (OpenMP), and GPU acceleration (CUDA) for efficient simulation.
- **CUDA Acceleration**: Offloads computationally intensive force calculations to GPUs, leveraging massive parallelism.
- **Non-blocking MPI Communication**: Optimizes data exchange between processes to overlap communication and computation.
- **Performance Profiling**: Analyzes bottlenecks using tools like `nvprof` and `Nsight Compute`.

---

## Visualization

### **1. Simulation Results**
The simulation output includes particle trajectories over time. Below is an example of a visualized trajectory generated from the simulation data:

![Particle Trajectories](images/particle_trajectories.png)

### **2. Performance Analysis**
To evaluate the optimization, performance charts are generated to show speedup and efficiency:

#### **Speedup vs. Number of Particles**
![Speedup](images/speedup_chart.png)

#### **GPU Utilization**
![GPU Utilization](images/gpu_utilization.png)

- **Insights**:
  - The CUDA-accelerated version shows significant performance improvement compared to the original implementation.
  - GPU utilization is maximized by optimizing memory access patterns and kernel execution.

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

---

## Implementation Details

### **1. Overall Architecture**
The program uses a hybrid parallel model:
- **MPI**: Distributes the simulation domain across multiple processes. Each MPI process is responsible for a specific spatial region.
- **OpenMP**: Parallelizes force calculations and other tasks within each MPI process using threads.
- **CUDA**: Offloads the most computationally intensive parts, such as force calculations, to the GPU.

### **2. Key Components**
- **Force Calculation**:
  - CUDA kernels are used to compute inter-particle forces, leveraging GPU's parallelism.
  - Optimized with shared memory to minimize global memory access latency.
- **Neighbor List**:
  - A neighbor list algorithm is implemented to reduce the \(O(N^2)\) complexity of force calculations by only considering nearby particles.
- **Communication**:
  - Non-blocking MPI (`MPI
