__global__ void compute_forces_kernel(float *positions, float *forces, int num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_particles)
    {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int j = 0; j < num_particles; j++)
        {
            if (i != j)
            {
                float dx = positions[j * 3] - positions[i * 3];
                float dy = positions[j * 3 + 1] - positions[i * 3 + 1];
                float dz = positions[j * 3 + 2] - positions[i * 3 + 2];
                float r2 = dx * dx + dy * dy + dz * dz;
                if (r2 > 0.0f)
                {
                    float r2_inv = 1.0f / r2;
                    float r6 = r2_inv * r2_inv * r2_inv;
                    float f = 24.0f * (2.0f * r6 * r6 - r6) * r2_inv;
                    fx += f * dx;
                    fy += f * dy;
                    fz += f * dz;
                }
            }
        }
        forces[i * 3] = fx;
        forces[i * 3 + 1] = fy;
        forces[i * 3 + 2] = fz;
    }
}
