#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

/**
 * Code due to Mahir Jain
 * Retrieved from https://github.com/mahirjain25/Parallel-Sorting-Algorithms
 * Adapted by Alexander Fischer <st149038@stud.uni-stuttgart.de>
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define BS_THREADS 1024

template <typename T>
__global__ void bitonic_sort_step(T *dev_values, int j, int k, uint64_t size) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size) {
        ixj = i ^ j;

        /* The threads with the lowest ids sort the array. */
        if ((ixj) > i) {
            if ((i & k) == 0) {
                /* Sort ascending */
                if (dev_values[i] > dev_values[ixj]) {
                    // swap
                    T temp = dev_values[i];
                    dev_values[i] = dev_values[ixj];
                    dev_values[ixj] = temp;
                }
            }
            if ((i & k) != 0) {
                /* Sort descending */
                if (dev_values[i] < dev_values[ixj]) {
                    // swap
                    T temp = dev_values[i];
                    dev_values[i] = dev_values[ixj];
                    dev_values[ixj] = temp;
                }
            }
        }
    }
}

// Added time tracking variables
template <typename T>
int bitonic_sort(uint64_t size, T *values, double &startTime, double &endTime) {
    T *dev_values;

    gpuErrchk(cudaMalloc(&dev_values, size * sizeof(T)));
    gpuErrchk(cudaMemcpy(dev_values, values, size * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    dim3 blocks(((size + BS_THREADS - 1) / BS_THREADS), 1);
    dim3 threads(BS_THREADS, 1);

    startTime = clock();

    /* Major step */
    for (uint64_t k = 2; k <= size; k <<= 1) {
        /* Minor step */
        for (uint64_t j = k >> 1; j > 0; j = j >> 1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k, size);
        }
    }
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    endTime = clock();

    gpuErrchk(cudaMemcpy(values, dev_values, size * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(dev_values);
    return 0;
}

#endif
