#ifndef MERGESORT_H
#define MERGESORT_H

/**
 * Code due to Mahir Jain
 * Retrieved from https://github.com/mahirjain25/Parallel-Sorting-Algorithms
 * Adapted by Rene Tischler <st149535@stud.uni-stuttgart.de>
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define MS_THREADS 1024
#define MS_BLOCKS 8

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x + threadIdx.y * (x = threads->x) + threadIdx.z * (x *= threads->y) + blockIdx.x * (x *= threads->z) + blockIdx.y * (x *= blocks->z) +
           blockIdx.z * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
template <typename T>
__device__ void gpu_bottomUpMerge(T* source, T* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

template <typename T>
__global__ void gpu_mergesort(T* source, T* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices, middle, end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size) break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
// Added time tracking variables & switched from long to int
template <typename T>
int mergesort(uint64_t size, T* data, double& startTime, double& endTime) {
    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    T* D_data;
    T* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    dim3 blocksPerGrid(MS_BLOCKS, 1);
    dim3 threadsPerBlock(MS_THREADS, 1);

    // Actually allocate the two arrays
    gpuErrchk(cudaMalloc((void**)&D_data, size * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&D_swp, size * sizeof(T)));

    // Copy from our input list into the first array
    gpuErrchk(cudaMemcpy(D_data, data, size * sizeof(T), cudaMemcpyHostToDevice));

    //
    // Copy the thread / block info to the GPU as well
    //
    gpuErrchk(cudaMalloc((void**)&D_threads, sizeof(dim3)));
    gpuErrchk(cudaMalloc((void**)&D_blocks, sizeof(dim3)));

    gpuErrchk(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    T* A = D_data;
    T* B = D_swp;

    startTime = clock();
    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (uint64_t width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads)*width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);
        gpuErrchk(cudaGetLastError());

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    gpuErrchk(cudaDeviceSynchronize());
    endTime = clock();
    gpuErrchk(cudaMemcpy(data, A, size * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(A);
    cudaFree(B);
    cudaFree(D_threads);
    cudaFree(D_blocks);
    return 0;
}

#endif
