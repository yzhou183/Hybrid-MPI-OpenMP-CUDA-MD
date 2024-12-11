#include "mpi_util.h"
#include "particle.h"

void distributeParticles(std::vector<Particle>& particles, int rank, int size, std::vector<Particle>& local_particles) {
    int num_particles_per_proc = particles.size() / size;
    int start_index = rank * num_particles_per_proc;
    int end_index = (rank == size - 1) ? particles.size() : start_index + num_particles_per_proc;

    for (int i = start_index; i < end_index; ++i) {
        local_particles.push_back(particles[i]);
    }
}
