#include <mpi.h>
#include "utils.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    initialize_simulation(); // 初始化
    for (int step = 0; step < total_steps; step++)
    {
        update_neighbor_list(); // 更新邻居列表
        compute_forces();       // 力计算
        update_positions();     // 更新位置
    }
    finalize_simulation(); // 释放资源

    MPI_Finalize();
    return 0;
}
