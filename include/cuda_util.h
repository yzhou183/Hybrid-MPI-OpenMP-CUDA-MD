#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cuda_runtime.h>
#include <vector>

void copyParticlesToDevice(const std::vector<Particle>& particles, Particle* d_particles);
void calculateForces(Particle* d_particles, int num_particles);
void copyParticlesFromDevice(Particle* d_particles, std::vector<Particle>& particles);

#endif // CUDA_UTIL_H
