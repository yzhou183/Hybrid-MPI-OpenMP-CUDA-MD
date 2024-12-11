#include <mpi.h>
#include <vector>
#include "particle.h"
#include "mpi_util.h"
#include "cuda_util.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int NUM_PARTICLES = 1024;
    std::vector<Particle> particles(NUM_PARTICLES);
    std::vector<Particle> local_particles;

    if (rank == 0) {
        initializeParticles(particles, NUM_PARTICLES);
    }

    distributeParticles(particles, rank, size, local_particles);

    Particle* d_particles;
    cudaMalloc(&d_particles, local_particles.size() * sizeof(Particle));
    copyParticlesToDevice(local_particles, d_particles);

    calculateForces(d_particles, local_particles.size());

    copyParticlesFromDevice(d_particles, local_particles);

    if (rank == 0) {
        printParticles(local_particles);
    }

    delete[] d_particles;

    MPI_Finalize();
    return 0;
}
