#ifndef ODD_EVEN_H
#define ODD_EVEN_H

/**
 * Code due to Mahir Jain
 * Retrieved from https://github.com/mahirjain25/Parallel-Sorting-Algorithms
 * Adapted by Alexander Fischer <st149038@stud.uni-stuttgart.de>
 */

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <ctime>

#define OS_BLOCKS 256
#define OS_THREADS 1024

template <typename T>
__global__ void odd_even_sort(T *c, int count) {
    for (int i = 0; i < ceil((double)count / 2); i++) {
        for (unsigned int grid = 0; grid * gridDim.x * blockDim.x < count; grid++) {
            unsigned int index = grid * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
            if ((!(index & 1)) && (index < count - 1))  // even phase
            {
                if (c[index] > c[index + 1]) swap(&c[index], &c[index + 1]);
            }
            __syncthreads();
            if ((index & 1) && (index < count - 1))  // odd phase
            {
                if (c[index] > c[index + 1]) swap(&c[index], &c[index + 1]);
            }
            __syncthreads();
        }
    }
}

// Added time tracking variables
template <typename T>
int odd_even_caller(uint64_t size, T *values, double &startTime, double &endTime) {
    T *dev_values;

    gpuErrchk(cudaMalloc((void **)&dev_values, size * sizeof(T)));
    gpuErrchk(cudaMemcpy(dev_values, values, size * sizeof(T), cudaMemcpyHostToDevice));

    unsigned int blocks = 1;
    unsigned int threads = std::min(size, (uint64_t)OS_THREADS);

    startTime = clock();
    odd_even_sort<<<blocks, threads>>>(dev_values, size);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaDeviceSynchronize());
    endTime = clock();

    gpuErrchk(cudaMemcpy(values, dev_values, size * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(dev_values);
    return 0;
}

#endif
