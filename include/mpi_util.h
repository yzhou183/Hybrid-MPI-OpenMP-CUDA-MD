#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include <mpi.h>
#include <vector>

void distributeParticles(std::vector<Particle>& particles, int rank, int size, std::vector<Particle>& local_particles);

#endif // MPI_UTIL_H
