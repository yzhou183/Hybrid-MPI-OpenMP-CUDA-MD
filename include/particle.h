#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

struct Particle {
    float x, y, z; 
    float fx, fy, fz;
};

void initializeParticles(std::vector<Particle>& particles, int num_particles);
void printParticles(const std::vector<Particle>& particles);

#endif
