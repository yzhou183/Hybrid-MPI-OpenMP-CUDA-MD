# Hybrid MPI + OpenMP + CUDA MD Program

This project focuses on enhancing the performence of Molecular Dynamics (MD) simulation program by leveraging a hybrid parallel computing model that integrates **MPI**, **OpenMP**, and **CUDA**. TThese simulations are computationally intensive, especially during force calculations, where the complexity grows quadratically with the number of particles.

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
  - Non-blocking MPI (`MPI_Isend` and `MPI_Irecv`) is used to exchange boundary data between processes while overlapping with local computations.
- **Performance Profiling**:
  - Tools like `nvprof` and `Nsight Compute` identify bottlenecks in kernel execution and data transfer.

### **3. Key Algorithms**
1. **Spatial Decomposition**:
   - The simulation domain is divided into subdomains assigned to different MPI processes.
   - Particles at subdomain boundaries are exchanged as "ghost particles" for accurate force calculation.

2. **CUDA Kernel for Force Calculation**:
   - Each thread computes forces for a single particle based on its neighbors.
   - Shared memory is used to store neighbor positions for faster access.

3. **Load Balancing**:
   - The number of particles per process is monitored to ensure even distribution and avoid idle processors.

---

## Challenges

1. **Performance Bottlenecks**:
   - Identifying and optimizing performance bottlenecks in CUDA kernels, MPI communication, and OpenMP threading.

2. **Data Management**:
   - Efficiently managing data transfer between host and device (CPU and GPU) to minimize overhead.
   - Ensuring data consistency across MPI processes and OpenMP threads.

3. **Load Balancing**:
   - Achieving an even distribution of computational workload across MPI processes and GPU threads, especially for non-uniform particle distributions.

4. **Debugging Parallel Code**:
   - Debugging issues like race conditions, synchronization errors, and memory leaks in a hybrid parallel environment.

5. **Scalability**:
   - Ensuring the program scales efficiently across multiple nodes and GPUs for large-scale simulations.

---

## Expected Results

1. **Performance Improvements**:
   - Significant reduction in force calculation time compared to the original MPI + OpenMP implementation.
   - Higher speedups achieved through CUDA acceleration and optimized kernel design.

2. **Scalability**:
   - Demonstration of strong and weak scaling for increasing system sizes and number of processes.

3. **Resource Utilization**:
   - Efficient use of GPU resources with high occupancy and minimal idle time.
   - Overlapping computation and communication to maximize throughput.

4. **Visualization and Analysis**:
   - Output simulation results and performance metrics for detailed analysis.
   - Generate performance charts (e.g., speedup vs. number of particles or processes).

5. **Extensibility**:
   - A modular and maintainable codebase that can be further extended to include more complex interactions or additional features.

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
