#include <gtest/gtest.h>
#include "../include/particle.h"

TEST(ParticleTest, Position) {
    Particle p(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(p.x, 1.0);
    EXPECT_DOUBLE_EQ(p.y, 2.0);
    EXPECT_DOUBLE_EQ(p.z, 3.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
