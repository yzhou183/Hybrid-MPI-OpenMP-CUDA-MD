#include "cuda_util.h"

__global__ void calculate_forces_kernel(Particle* d_particles, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    Particle p = d_particles[idx];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int j = 0; j < num_particles; ++j) {
        if (idx != j) {
            float dx = d_particles[j].x - p.x;
            float dy = d_particles[j].y - p.y;
            float dz = d_particles[j].z - p.z;
            float r2 = dx * dx + dy * dy + dz * dz;
            float inv_r3 = 1.0f / (r2 * sqrtf(r2));
            fx += dx * inv_r3;
            fy += dy * inv_r3;
            fz += dz * inv_r3;
        }
    }

    d_particles[idx].fx = fx;
    d_particles[idx].fy = fy;
    d_particles[idx].fz = fz;
}

void copyParticlesToDevice(const std::vector<Particle>& particles, Particle* d_particles) {
    cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);
}

void calculateForces(Particle* d_particles, int num_particles) {
    dim3 blockSize(256);
    dim3 gridSize((num_particles + blockSize.x - 1) / blockSize.x);
    calculate_forces_kernel<<<gridSize, blockSize>>>(d_particles, num_particles);
    cudaDeviceSynchronize();
}

void copyParticlesFromDevice(Particle* d_particles, std::vector<Particle>& particles) {
    cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);
}
