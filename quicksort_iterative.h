#ifndef QUICKSORT_ITERATIVE_H
#define QUICKSORT_ITERATIVE_H

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <string>

#define QS_BLOCKS 8
#define QS_THREADS 1024

// iterative partition (hoare scheme)
template <typename T>
__global__ void partition_iterative(unsigned long long int partitions, uint64_t *cuts, uint64_t *newCuts, unsigned long long int *cutCounter, bool *isFinished, T *data) {
    unsigned long long int partitionIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; partitionIdx < partitions; partitionIdx += gridDim.x * blockDim.x) {
        uint64_t l = cuts[2 * partitionIdx];
        uint64_t r = cuts[2 * partitionIdx + 1];

        uint64_t middle = (l + r) / 2;

        // Choose pivot (median of three)
        if (data[l] > data[middle]) {
            swap(&data[l], &data[middle]);
        }
        if (data[l] > data[r]) {
            swap(&data[l], &data[r]);
        }
        if (data[middle] > data[r]) {
            swap(&data[middle], &data[r]);
        }

        if (r - l < 2) {
            continue;
        }

        T pivot = data[middle];
        uint64_t i = l;
        uint64_t j = r;
        while (true) {
            do {
                i++;
            } while (data[i] < pivot);
            do {
                j--;
            } while (data[j] > pivot);
            if (i >= j) {
                // printf("%ld-%ld:%ld(%i)\n", l, r, j + 1, partitionIdx);
                if (j - l > 0 || r - j - 1 > 0) {
                    *isFinished = false;
                    if (j - l > 0) {
                        unsigned long long int index = atomicAdd(cutCounter, 2);
                        newCuts[index] = l;
                        newCuts[index + 1] = j;
                    }
                    if (r - j - 1 > 0) {
                        unsigned long long int index = atomicAdd(cutCounter, 2);
                        newCuts[index] = j + 1;
                        newCuts[index + 1] = r;
                    }
                }
                goto nextPartition;
            }
            swap(&data[i], &data[j]);
        }
    nextPartition:;
    }
}

// function to copy memory and start iterative quicksort
template <typename T>
int quicksort_iterative(uint64_t n, T *h_data, int verbose, double &startTime, double &endTime) {
    T *d_data;
    bool *d_isFinished;
    uint64_t *d_cuts;
    uint64_t *d_newCuts;
    unsigned long long int *d_cutCounter;

    if (verbose) {
        std::cout << "Copying data to device..." << std::endl;
    }
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_cuts, 2 * n * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&d_newCuts, 2 * n * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&d_cutCounter, sizeof(unsigned long long int)));
    gpuErrchk(cudaMalloc(&d_isFinished, sizeof(bool)));

    uint64_t *h_cuts;
    h_cuts = (uint64_t *)malloc(2 * n * sizeof(uint64_t));
    h_cuts[0] = 0;
    h_cuts[1] = n - 1;
    unsigned long long int h_cutCounter = 2;
    unsigned long long int newCutCounter = 0;
    bool h_isFinished;
    gpuErrchk(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cuts, h_cuts, 2 * n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cutCounter, &h_cutCounter, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    if (verbose) {
        std::cout << "Sorting data..." << std::endl;
    }
    startTime = clock();

    do {
        // TODO: copy memory async?
        h_isFinished = true;
        gpuErrchk(cudaMemcpy(d_isFinished, &h_isFinished, sizeof(bool), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_cutCounter, &newCutCounter, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaDeviceSynchronize());

        unsigned long long int partitions = h_cutCounter / 2;
        unsigned int blocks = std::min((int)ceil((double)partitions / QS_THREADS), QS_BLOCKS);
        unsigned int threads = std::min(partitions, (unsigned long long int)QS_THREADS);
        // std::cout << blocks * threads << "/" << partitions << " threads starting" << std::endl;
        partition_iterative<<<blocks, threads>>>(partitions, d_cuts, d_newCuts, d_cutCounter, d_isFinished, d_data);
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(&h_isFinished, d_isFinished, sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&h_cutCounter, d_cutCounter, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(d_cuts, d_newCuts, 2 * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();
        // std::cout << "isFinished = " << h_isFinished << std::endl;
    } while (!h_isFinished);

    gpuErrchk(cudaDeviceSynchronize());
    endTime = clock();

    if (verbose) {
        std::cout << "Copying data to host..." << std::endl;
    }
    gpuErrchk(cudaMemcpy(h_data, d_data, n * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_isFinished);
    cudaFree(d_cuts);
    cudaFree(d_newCuts);
    cudaFree(d_cutCounter);
    free(h_cuts);
    return 0;
}

#endif
