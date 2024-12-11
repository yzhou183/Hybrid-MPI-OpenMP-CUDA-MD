#include "particle.h"

void initializeParticles(std::vector<Particle>& particles, int num_particles) {
    for (int i = 0; i < num_particles; ++i) {
        particles[i].x = static_cast<float>(i) / num_particles;
        particles[i].y = static_cast<float>(i) * 0.1f;
        particles[i].z = static_cast<float>(i) * 0.2f;
    }
}

void printParticles(const std::vector<Particle>& particles) {
    for (const auto& p : particles) {
        std::cout << "Particle: (" << p.x << ", " << p.y << ", " << p.z << ") -> ("
                  << p.fx << ", " << p.fy << ", " << p.fz << ")" << std::endl;
    }
}
