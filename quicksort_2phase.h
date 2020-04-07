#ifndef QUICKSORT_2PHASE_H
#define QUICKSORT_2PHASE_H

/**
 * Algorithm due to Daniel Cederman & Philippas Tsigas
 * Based on http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.156.9622&rep=rep1&type=pdf
 * Implemented by Alexander Fischer
 */

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "helper.h"

#define MAX_BLOCKS 256
#define QS2_THREADS 1024
#define ELEMENTS_PER_QS2_THREAD 64  // 64 is best?
#define THRESHOLD 1

template <typename T>
__global__ void quicksort_pivot(T *data, T *pivot, unsigned long long *partitions, unsigned long long *partitionSizes) {
    for (unsigned long long partition = blockIdx.x * blockDim.x + threadIdx.x; partition < partitionSizes[0]; partition += gridDim.x * blockDim.x) {
        // Median of three
        unsigned long long right = partitions[partition] + partitionSizes[partition + 1] - 1;
        T first = data[partitions[partition]];
        T second = data[(partitions[partition] + right) / 2];
        T third = data[right];
        if ((first > second) != (first > third)) {
            pivot[partition] = first;
        } else if ((second > first) != (second > third)) {
            pivot[partition] = second;
        } else {
            pivot[partition] = third;
        }
    }
}

template <typename T>
__global__ void quicksort_createPartitions(T *currentData, T *newData, unsigned long long *pivotOffsets, unsigned long long *greaterOffsets,
                                           unsigned long long *currentPartitions, unsigned long long *newPartitions, unsigned long long *currentPartitionSizes,
                                           unsigned long long *newPartitionSizes, unsigned long long threshold) {
    unsigned long long partitionCount = currentPartitionSizes[0];

    if (threadIdx.x == 0) {
        newPartitionSizes[0] = 0;
        currentPartitionSizes[0] = 0;
    }
    __syncthreads();

    for (unsigned long long partition = threadIdx.x; partition < partitionCount; partition += blockDim.x) {
        if (pivotOffsets[partition] <= threshold) {
            for (unsigned long long i = currentPartitions[partition]; i < currentPartitions[partition] + pivotOffsets[partition]; i++) {
                currentData[i] = newData[i];
            }
        } else {
            unsigned long long index = atomicAdd(&newPartitionSizes[0], 1);
            newPartitions[index] = currentPartitions[partition];
            newPartitionSizes[index + 1] = pivotOffsets[partition];
        }
        if (currentPartitionSizes[partition + 1] - greaterOffsets[partition] <= threshold) {
            for (unsigned long long i = currentPartitions[partition] + greaterOffsets[partition]; i < currentPartitions[partition] + currentPartitionSizes[partition + 1];
                 i++) {
                currentData[i] = newData[i];
            }
        } else {
            unsigned long long index = atomicAdd(&newPartitionSizes[0], 1);
            newPartitions[index] = currentPartitions[partition] + greaterOffsets[partition];
            newPartitionSizes[index + 1] = currentPartitionSizes[partition + 1] - greaterOffsets[partition];
        }

        for (unsigned long long i = currentPartitions[partition] + pivotOffsets[partition]; i < currentPartitions[partition] + greaterOffsets[partition]; i++) {
            currentData[i] = newData[i];
        }
    }
}

// Doesn't work on 1500+ partitions, due to the 48KB shared memory limit
// Kernel to create new partitions and write finished partitions into currentData (executes in a single block to use shared memory)
template <typename T>
__global__ void quicksort_createPartitions_shared(T *currentData, T *newData, unsigned long long *pivotOffsets, unsigned long long *greaterOffsets,
                                                  unsigned long long *currentPartitions, unsigned long long *newPartitions, unsigned long long *currentPartitionSizes,
                                                  unsigned long long *newPartitionSizes, unsigned long long threshold) {
    extern __shared__ unsigned long long s[];
    unsigned long long *s_newPartitions = s;
    unsigned long long *s_newPartitionSizes = &s[2 * currentPartitionSizes[0]];

    unsigned long long partitionCount = currentPartitionSizes[0];

    if (threadIdx.x == 0) {
        s_newPartitionSizes[0] = 0;
    }
    __syncthreads();

    for (unsigned long long partition = threadIdx.x; partition < partitionCount; partition += blockDim.x) {
        if (pivotOffsets[partition] <= threshold) {
            for (unsigned long long i = currentPartitions[partition]; i < currentPartitions[partition] + pivotOffsets[partition]; i++) {
                currentData[i] = newData[i];
            }
        } else {
            unsigned long long index = atomicAdd(&s_newPartitionSizes[0], 1);
            s_newPartitions[index] = currentPartitions[partition];
            s_newPartitionSizes[index + 1] = pivotOffsets[partition];
        }
        if (currentPartitionSizes[partition + 1] - greaterOffsets[partition] <= threshold) {
            for (unsigned long long i = currentPartitions[partition] + greaterOffsets[partition]; i < currentPartitions[partition] + currentPartitionSizes[partition + 1];
                 i++) {
                currentData[i] = newData[i];
            }
        } else {
            unsigned long long index = atomicAdd(&s_newPartitionSizes[0], 1);
            s_newPartitions[index] = currentPartitions[partition] + greaterOffsets[partition];
            s_newPartitionSizes[index + 1] = currentPartitionSizes[partition + 1] - greaterOffsets[partition];
        }

        for (unsigned long long i = currentPartitions[partition] + pivotOffsets[partition]; i < currentPartitions[partition] + greaterOffsets[partition]; i++) {
            currentData[i] = newData[i];
        }
    }

    __syncthreads();

    for (unsigned long long newPartition = threadIdx.x; newPartition < s_newPartitionSizes[0]; newPartition += blockDim.x) {
        newPartitions[newPartition] = s_newPartitions[newPartition];
        newPartitionSizes[newPartition + 1] = s_newPartitionSizes[newPartition + 1];
    }
    if (threadIdx.x == 0) {
        newPartitionSizes[0] = s_newPartitionSizes[0];
        currentPartitionSizes[0] = 0;
    }
}

template <typename T>
__global__ void quicksort_allSingleThreaded(T *currentData, T *newData, T *pivots, unsigned long long *pivotOffsets, unsigned long long *greaterOffsets,
                                            unsigned long long *singleThreadPartitions, unsigned long long singleThreadCount, unsigned long long *currentPartitions,
                                            unsigned long long *currentPartitionSizes) {
    for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x; index < singleThreadCount; index += gridDim.x * blockDim.x) {
        unsigned long long partition = singleThreadPartitions[index];
        T *partitionData = &currentData[currentPartitions[partition]];
        T *newPartitionData = &newData[currentPartitions[partition]];
        unsigned long long partitionSize = currentPartitionSizes[partition + 1];
        unsigned long long smallerCounter = 0;
        unsigned long long greaterCounter = 0;
        unsigned long long pivotCounter = 0;
        T pivot = pivots[partition];

        // Count/Move elements
        for (unsigned long long i = 0; i < partitionSize; i++) {
            if (partitionData[i] < pivot) {
                newPartitionData[smallerCounter++] = partitionData[i];
            } else if (partitionData[i] > pivot) {
                newPartitionData[partitionSize - 1 - greaterCounter++] = partitionData[i];
            }
        }
        for (unsigned long long i = 0; i < partitionSize; i++) {
            if (partitionData[i] == pivot) {
                newPartitionData[smallerCounter + pivotCounter++] = partitionData[i];
            }
        }
        assert(pivotCounter > 0);

        pivotOffsets[partition] = smallerCounter;
        greaterOffsets[partition] = smallerCounter + pivotCounter;
    }
}

template <typename T>
__global__ void quicksort_allMultiThreaded1(T *currentData, T *newData, T *pivots, unsigned long long *multiThreadPartitions,
                                            unsigned long long *multiThreadPartitionMembers, unsigned long long multiThreadCount,
                                            unsigned long long *firstPartitionIndexes, unsigned long long *currentPartitions, unsigned long long *currentPartitionSizes,
                                            unsigned long long *tSmaller, unsigned long long *tGreater, unsigned long long *tPivot, unsigned long long elementsThread) {
    for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x; index < multiThreadCount; index += gridDim.x * blockDim.x) {
        unsigned long long partition = multiThreadPartitions[index];
        unsigned long long partitionMember = multiThreadPartitionMembers[index];
        T *partitionData = &currentData[currentPartitions[partition]];
        unsigned long long partitionSize = currentPartitionSizes[partition + 1];
        unsigned long long memberSize = ceil((double)partitionSize / elementsThread);
        T pivot = pivots[partition];

        tSmaller[index] = 0;
        tGreater[index] = 0;
        tPivot[index] = 0;
        for (unsigned long long i = partitionMember; i < partitionSize; i += memberSize) {
            if (partitionData[i] < pivot) {
                tSmaller[index]++;
            } else if (partitionData[i] > pivot) {
                tGreater[index]++;
            } else {
                tPivot[index]++;
            }
        }
        if (partitionMember == 0) {
            firstPartitionIndexes[partition] = index;
        }
    }
}

#define MAX_SCAN_BLOCKS 1024
#define SCAN_THREADS 128
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
__global__ void scanBlocks(unsigned long long *data, unsigned long long size, bool writeResults, unsigned long long *blockSums) {
    extern __shared__ unsigned long long temp[];
    int index = threadIdx.x;
    for (unsigned int block = blockIdx.x; block < size / SCAN_THREADS; block += gridDim.x) {
        int blockOffset = block * SCAN_THREADS;
        int offset = 1;

        // load input into shared memory
        int ai = index;
        int bi = index + (SCAN_THREADS / 2);
        int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
        temp[ai + bankOffsetA] = data[blockOffset + ai];
        temp[bi + bankOffsetB] = data[blockOffset + bi];

        // build sum in place up the tree
        for (int d = SCAN_THREADS >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (index < d) {
                int ai = offset * (2 * index + 1) - 1;
                int bi = offset * (2 * index + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                temp[bi] += temp[ai];
            }
            offset *= 2;
        }

        // clear the last element and store blockSum if multiple blocks
        if (index == 0) {
            if (writeResults) {
                blockSums[block] = temp[SCAN_THREADS - 1 + CONFLICT_FREE_OFFSET(SCAN_THREADS - 1)];
            }
            temp[SCAN_THREADS - 1 + CONFLICT_FREE_OFFSET(SCAN_THREADS - 1)] = 0;
        }

        // traverse down tree & build scan
        for (int d = 1; d < SCAN_THREADS; d *= 2) {
            offset >>= 1;
            __syncthreads();
            if (index < d) {
                int ai = offset * (2 * index + 1) - 1;
                int bi = offset * (2 * index + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        __syncthreads();

        // write results to device memory
        data[blockOffset + ai] = temp[ai + bankOffsetA];
        data[blockOffset + bi] = temp[bi + bankOffsetB];

        __syncthreads();
    }
}

// Call with one block less
__global__ void sumBlocks(unsigned long long *data, unsigned long long size, unsigned long long *outerOffsets, unsigned long long *gridOffsets,
                          unsigned long long *blockOffsets) {
    for (unsigned long long outer = 0; outer * SCAN_THREADS * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x < size; outer++) {
        for (unsigned long long grid = 0; grid * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x < size && grid < SCAN_THREADS; grid++) {
            data[(outer * SCAN_THREADS + grid) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x] +=
                outerOffsets[outer] + gridOffsets[outer * SCAN_THREADS + grid] + blockOffsets[outer * SCAN_THREADS * gridDim.x + grid * gridDim.x + blockIdx.x];
        }
    }
}

__global__ void quicksort_allMultiThreaded2(unsigned long long *pivotOffsets, unsigned long long *greaterOffsets, unsigned long long multiThreadCount,
                                            unsigned long long *firstPartitionIndexes, unsigned long long *currentPartitions, unsigned long long *currentPartitionSizes,
                                            unsigned long long *tSmaller, unsigned long long *tGreater, unsigned long long *tPivot, unsigned long long elementsThread) {
    for (unsigned long long partition = blockIdx.x * blockDim.x + threadIdx.x; partition < currentPartitionSizes[0]; partition += gridDim.x * blockDim.x) {
        if (currentPartitionSizes[partition + 1] > elementsThread) {
            unsigned long long firstIndex = firstPartitionIndexes[partition];
            unsigned long long memberSize = ceil((double)currentPartitionSizes[partition + 1] / elementsThread);

            unsigned long long lastSmaller = tSmaller[firstIndex];
            unsigned long long lastGreater = tGreater[firstIndex];
            unsigned long long lastPivot = tPivot[firstIndex];
            unsigned long long tempSmaller;
            unsigned long long tempGreater;
            unsigned long long tempPivot;
            tSmaller[firstIndex] = 0;
            tGreater[firstIndex] = 0;
            tPivot[firstIndex] = 0;
            for (unsigned long long partitionMember = firstIndex + 1; partitionMember < firstIndex + memberSize; partitionMember++) {
                tempSmaller = tSmaller[partitionMember];
                tempGreater = tGreater[partitionMember];
                tempPivot = tPivot[partitionMember];
                tSmaller[partitionMember] = lastSmaller + tSmaller[partitionMember - 1];
                tGreater[partitionMember] = lastGreater + tGreater[partitionMember - 1];
                tPivot[partitionMember] = lastPivot + tPivot[partitionMember - 1];
                lastSmaller = tempSmaller;
                lastGreater = tempGreater;
                lastPivot = tempPivot;
            }

            unsigned long long pOffset = tSmaller[firstIndex + memberSize - 1] + lastSmaller;
            pivotOffsets[partition] = pOffset;
            greaterOffsets[partition] = pOffset + tPivot[firstIndex + memberSize - 1] + lastPivot;
        }
    }
}

template <typename T>
__global__ void quicksort_allMultiThreaded3(T *currentData, T *newData, T *pivots, unsigned long long *pivotOffsets, unsigned long long *greaterOffsets,
                                            unsigned long long *multiThreadPartitions, unsigned long long *multiThreadPartitionMembers,
                                            unsigned long long multiThreadCount, unsigned long long *currentPartitions, unsigned long long *currentPartitionSizes,
                                            unsigned long long *tSmaller, unsigned long long *tGreater, unsigned long long *tPivot, unsigned long long elementsThread) {
    for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x; index < multiThreadCount; index += gridDim.x * blockDim.x) {
        unsigned long long partition = multiThreadPartitions[index];
        unsigned long long partitionMember = multiThreadPartitionMembers[index];
        T *partitionData = &currentData[currentPartitions[partition]];
        T *newPartitionData = &newData[currentPartitions[partition]];
        unsigned long long partitionSize = currentPartitionSizes[partition + 1];
        unsigned long long memberSize = ceil((double)partitionSize / elementsThread);
        T pivot = pivots[partition];

        unsigned long long smallerCounter = 0;
        unsigned long long greaterCounter = 0;
        unsigned long long pivotCounter = 0;
        for (unsigned long long i = partitionMember; i < partitionSize; i += memberSize) {
            if (partitionData[i] < pivot) {
                newPartitionData[tSmaller[index] + smallerCounter] = partitionData[i];
                smallerCounter++;
            } else if (partitionData[i] > pivot) {
                newPartitionData[greaterOffsets[partition] + tGreater[index] + greaterCounter++] = partitionData[i];
            } else {
                newPartitionData[pivotOffsets[partition] + tPivot[index] + pivotCounter++] = partitionData[i];
            }
        }
    }
}

template <typename T>
int quicksort_2phase(unsigned long long size, T *h_data, double &startTime, double &endTime, unsigned long long threshold, int maxBlocks = 0, int qs2Threads = 0,
                     unsigned long long elementsThread = 0) {
    if (maxBlocks == 0) {
        maxBlocks = 256;
    }
    if (qs2Threads == 0) {
        qs2Threads = 1024;
    }
    if (elementsThread == 0) {
        elementsThread = 64;
    }

    T *d_data1, *d_data2, *d_currentData, *d_newData, *d_pivot;
    // Partition handling, all partitionSizes-arrays contain size in first element
    unsigned long long *d_pivotOffsets, *d_greaterOffsets, *d_partitions1, *d_partitions2, *d_currentPartitions, *d_newPartitions, *d_partitionSizes1, *d_partitionSizes2,
        *d_currentPartitionSizes, *d_newPartitionSizes, *h_partitionSizes;
    // Arrays for numbers of elements greater/smaller than the pivot (threads)
    unsigned long long *d_tSmaller, *d_tGreater, *d_tPivot;
    // Arrays for cumulative sums (blocks)
    unsigned long long *d_bSmaller, *d_bGreater, *d_bPivot;
    // Arrays to handle bulk kernel
    unsigned long long *d_singleThreadPartitions, *h_singleThreadPartitions;
    unsigned long long *d_multiThreadPartitions, *d_multiThreadPartitionMembers, *h_multiThreadPartitions, *h_multiThreadPartitionMembers;
    unsigned long long *d_firstPartitionIndexes;

    cudaStream_t singleThreadStream;
    cudaStream_t multiThreadStream;
    cudaStreamCreate(&singleThreadStream);
    cudaStreamCreate(&multiThreadStream);

    unsigned long long thresholdSize = std::ceil((double)size / (threshold + 1));
    unsigned long long zero = 0;

    gpuErrchk(cudaMallocHost(&h_partitionSizes, (thresholdSize + 1) * sizeof(unsigned long long)));
    gpuErrchk(cudaMallocHost(&h_singleThreadPartitions, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMallocHost(&h_multiThreadPartitions, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMallocHost(&h_multiThreadPartitionMembers, thresholdSize * sizeof(unsigned long long)));

    unsigned long long *blocks = new unsigned long long[thresholdSize];
    unsigned long long *threads = new unsigned long long[thresholdSize];
    unsigned long long *threadOffsets = new unsigned long long[thresholdSize];
    unsigned long long *blockOffsets = new unsigned long long[thresholdSize];

    gpuErrchk(cudaMalloc(&d_data1, size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_data2, size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_pivot, thresholdSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_pivotOffsets, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_greaterOffsets, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_partitions1, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_partitions2, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_partitionSizes1, (thresholdSize + 1) * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_partitionSizes2, (thresholdSize + 1) * sizeof(unsigned long long)));

    gpuErrchk(cudaMalloc(&d_tSmaller, 2 * thresholdSize * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_tGreater, 2 * thresholdSize * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_tPivot, 2 * thresholdSize * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_bSmaller, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_bGreater, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_bPivot, thresholdSize * sizeof(unsigned long long)));

    gpuErrchk(cudaMalloc(&d_singleThreadPartitions, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_multiThreadPartitions, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_multiThreadPartitionMembers, thresholdSize * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&d_firstPartitionIndexes, thresholdSize * sizeof(unsigned long long)));

    gpuErrchk(cudaMemcpy(d_data1, h_data, size * sizeof(T), cudaMemcpyHostToDevice));

    gpuErrchk(cudaDeviceSynchronize());
    startTime = clock();

    d_currentData = d_data1;
    d_newData = d_data2;
    d_currentPartitions = d_partitions1;
    d_newPartitions = d_partitions2;
    d_currentPartitionSizes = d_partitionSizes1;
    d_newPartitionSizes = d_partitionSizes2;

    gpuErrchk(cudaMemcpy(d_currentPartitions, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    h_partitionSizes[0] = 1;
    h_partitionSizes[1] = size;
    gpuErrchk(cudaMemcpy(d_currentPartitionSizes, h_partitionSizes, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    do {
        // Put partitions into single/multi thread arrays
        unsigned long long partitionCount = h_partitionSizes[0];
        unsigned long long singleThreadCount = 0;
        unsigned long long multiThreadCount = 0;
        for (unsigned long long partition = 0; partition < partitionCount; partition++) {
            if (h_partitionSizes[partition + 1] > elementsThread) {
                for (unsigned long long partitionMember = 0; partitionMember < std::ceil((double)h_partitionSizes[partition + 1] / elementsThread); partitionMember++) {
                    h_multiThreadPartitions[multiThreadCount] = partition;
                    h_multiThreadPartitionMembers[multiThreadCount] = partitionMember;
                    multiThreadCount++;
                }
            } else {
                h_singleThreadPartitions[singleThreadCount++] = partition;
            }
        }
        gpuErrchk(cudaMemcpy(d_singleThreadPartitions, h_singleThreadPartitions, singleThreadCount * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_multiThreadPartitions, h_multiThreadPartitions, multiThreadCount * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_multiThreadPartitionMembers, h_multiThreadPartitionMembers, multiThreadCount * sizeof(unsigned long long), cudaMemcpyHostToDevice));

        // Calculate pivot
        int pBlocks = std::min((int)std::ceil((double)partitionCount / qs2Threads), maxBlocks);
        int pThreads = std::min(partitionCount, (unsigned long long)qs2Threads);
        quicksort_pivot<<<pBlocks, pThreads>>>(d_currentData, d_pivot, d_currentPartitions, d_currentPartitionSizes);

        // Sort single thread partitions
        if (singleThreadCount > 0) {
            int blocks = std::min((int)std::ceil((double)singleThreadCount / qs2Threads), maxBlocks);
            int threads = std::min(singleThreadCount, (unsigned long long)qs2Threads);
            quicksort_allSingleThreaded<<<blocks, threads, 0, singleThreadStream>>>(d_currentData, d_newData, d_pivot, d_pivotOffsets, d_greaterOffsets,
                                                                                    d_singleThreadPartitions, singleThreadCount, d_currentPartitions,
                                                                                    d_currentPartitionSizes);
        }

        // Sort multi thread partitions
        if (multiThreadCount > 0) {
            int blocks = std::min((int)std::ceil((double)multiThreadCount / qs2Threads), maxBlocks);
            int threads = std::min(multiThreadCount, (unsigned long long)qs2Threads);
            int countBlocks = std::min((int)std::ceil((double)partitionCount / qs2Threads), maxBlocks);
            int countThreads = std::min(partitionCount, (unsigned long long)qs2Threads);
            quicksort_allMultiThreaded1<<<blocks, threads, 0, multiThreadStream>>>(
                d_currentData, d_newData, d_pivot, d_multiThreadPartitions, d_multiThreadPartitionMembers, multiThreadCount, d_firstPartitionIndexes, d_currentPartitions,
                d_currentPartitionSizes, d_tSmaller, d_tGreater, d_tPivot, elementsThread);
            quicksort_allMultiThreaded2<<<countBlocks, countThreads, 0, multiThreadStream>>>(d_pivotOffsets, d_greaterOffsets, multiThreadCount, d_firstPartitionIndexes,
                                                                                             d_currentPartitions, d_currentPartitionSizes, d_tSmaller, d_tGreater,
                                                                                             d_tPivot, elementsThread);
            quicksort_allMultiThreaded3<<<blocks, threads, 0, multiThreadStream>>>(
                d_currentData, d_newData, d_pivot, d_pivotOffsets, d_greaterOffsets, d_multiThreadPartitions, d_multiThreadPartitionMembers, multiThreadCount,
                d_currentPartitions, d_currentPartitionSizes, d_tSmaller, d_tGreater, d_tPivot, elementsThread);
        }

        // Create new partitions
        if (partitionCount < 1450) {
            quicksort_createPartitions_shared<<<1, std::min(partitionCount, (unsigned long long)qs2Threads), (4 * partitionCount + 1) * sizeof(unsigned long long)>>>(
                d_currentData, d_newData, d_pivotOffsets, d_greaterOffsets, d_currentPartitions, d_newPartitions, d_currentPartitionSizes, d_newPartitionSizes,
                threshold);
        } else {
            quicksort_createPartitions<<<1, std::min(partitionCount, (unsigned long long)qs2Threads)>>>(d_currentData, d_newData, d_pivotOffsets, d_greaterOffsets,
                                                                                                        d_currentPartitions, d_newPartitions, d_currentPartitionSizes,
                                                                                                        d_newPartitionSizes, threshold);
        }

        gpuErrchk(
            cudaMemcpy(h_partitionSizes, d_newPartitionSizes, std::min(2 * partitionCount + 1, thresholdSize) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());

        // Swap pointers
        T *tempData = d_currentData;
        d_currentData = d_newData;
        d_newData = tempData;

        unsigned long long *tempPartitions = d_currentPartitions;
        d_currentPartitions = d_newPartitions;
        d_newPartitions = tempPartitions;

        unsigned long long *tempPartitionSizes = d_currentPartitionSizes;
        d_currentPartitionSizes = d_newPartitionSizes;
        d_newPartitionSizes = tempPartitionSizes;
    } while (h_partitionSizes[0] > 0);

    gpuErrchk(cudaDeviceSynchronize());
    endTime = clock();

    gpuErrchk(cudaMemcpy(h_data, d_currentData, size * sizeof(T), cudaMemcpyDeviceToHost));

    cudaStreamDestroy(singleThreadStream);
    cudaStreamDestroy(multiThreadStream);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_pivot);
    cudaFree(d_partitions1);
    cudaFree(d_partitions2);
    cudaFree(d_partitionSizes1);
    cudaFree(d_partitionSizes2);
    cudaFree(d_pivotOffsets);
    cudaFree(d_greaterOffsets);
    cudaFree(d_tSmaller);
    cudaFree(d_tGreater);
    cudaFree(d_tPivot);
    cudaFree(d_bSmaller);
    cudaFree(d_bGreater);
    cudaFree(d_bPivot);
    cudaFree(d_singleThreadPartitions);
    cudaFree(d_multiThreadPartitions);
    cudaFree(d_multiThreadPartitionMembers);
    cudaFree(d_firstPartitionIndexes);
    cudaFreeHost(h_partitionSizes);
    cudaFreeHost(h_singleThreadPartitions);
    cudaFreeHost(h_multiThreadPartitions);
    cudaFreeHost(h_multiThreadPartitionMembers);
    delete[] threadOffsets;
    delete[] blockOffsets;
    return 0;
}

//
// Old unused kernels
//

template <typename T>
__global__ void quicksort_phase1(unsigned int blocks, T *currentData, T *pivot, unsigned long long *currentPartition, unsigned long long *currentPartitionSize,
                                 unsigned int *tSmaller, unsigned int *tGreater, unsigned int *tPivot, unsigned long long elementsThread) {
    T *partitionData = &currentData[*currentPartition];
    unsigned long long partitionSize = *currentPartitionSize;
    for (unsigned long long block = blockIdx.x; block < blocks; block += gridDim.x) {
        unsigned long long nextIndex = block * blockDim.x + ((threadIdx.x + 1) % blockDim.x);
        tSmaller[nextIndex] = 0;
        tGreater[nextIndex] = 0;
        tPivot[nextIndex] = 0;
        for (unsigned long long i = block * blockDim.x * elementsThread + threadIdx.x; i < partitionSize && i < (block + 1) * blockDim.x * elementsThread;
             i += blockDim.x) {
            if (partitionData[i] < *pivot) {
                tSmaller[nextIndex]++;
            } else if (partitionData[i] > *pivot) {
                tGreater[nextIndex]++;
            } else {
                tPivot[nextIndex]++;
            }
        }
    }
}

__global__ void quicksort_calculate_thread_offsets(unsigned int blocks, unsigned int threads, unsigned int *tSmaller, unsigned int *tGreater, unsigned int *tPivot) {
    for (unsigned long long block = blockIdx.x * blockDim.x + threadIdx.x; block < blocks; block += gridDim.x * blockDim.x) {
        for (unsigned int thread = block * threads + 2; thread < (block + 1) * threads; thread++) {
            tSmaller[thread] += tSmaller[thread - 1];
            tGreater[thread] += tGreater[thread - 1];
            tPivot[thread] += tPivot[thread - 1];
        }
    }
}

__global__ void quicksort_calculate_block_offsets(unsigned long long *pivotOffset, unsigned long long *greaterOffset, unsigned int blocks, unsigned int threads,
                                                  unsigned int *tSmaller, unsigned int *tGreater, unsigned int *tPivot, unsigned long long *bSmaller,
                                                  unsigned long long *bGreater, unsigned long long *bPivot) {
    bSmaller[0] = 0;
    bGreater[0] = 0;
    bPivot[0] = 0;
    for (unsigned int block = 1; block < blocks; block++) {
        bSmaller[block] = bSmaller[block - 1] + tSmaller[block * threads - 1] + tSmaller[(block - 1) * threads];
        bGreater[block] = bGreater[block - 1] + tGreater[block * threads - 1] + tGreater[(block - 1) * threads];
        bPivot[block] = bPivot[block - 1] + tPivot[block * threads - 1] + tPivot[(block - 1) * threads];
        tSmaller[(block - 1) * threads] = 0;
        tGreater[(block - 1) * threads] = 0;
        tPivot[(block - 1) * threads] = 0;
    }
    unsigned long long pOffset = bSmaller[blocks - 1] + tSmaller[blocks * threads - 1] + tSmaller[(blocks - 1) * threads];
    *pivotOffset = pOffset;
    *greaterOffset = pOffset + bPivot[blocks - 1] + tPivot[blocks * threads - 1] + tPivot[(blocks - 1) * threads];
    assert(*pivotOffset < *greaterOffset);

    tSmaller[(blocks - 1) * threads] = 0;
    tGreater[(blocks - 1) * threads] = 0;
    tPivot[(blocks - 1) * threads] = 0;
}

template <typename T>
__global__ void quicksort_phase2(unsigned int blocks, T *currentData, T *newData, T *pivot, unsigned long long *pivotOffset, unsigned long long *greaterOffset,
                                 unsigned long long *currentPartition, unsigned long long *currentPartitionSize, unsigned long long *newPartition,
                                 unsigned long long *newPartitionSize, unsigned int *tSmaller, unsigned int *tGreater, unsigned int *tPivot, unsigned long long *bSmaller,
                                 unsigned long long *bGreater, unsigned long long *bPivot, unsigned long long elementsThread) {
    T *partitionData = &currentData[*currentPartition];
    T *newPartitionData = &newData[*currentPartition];
    unsigned long long partitionSize = *currentPartitionSize;
    for (unsigned long long block = blockIdx.x; block < blocks; block += gridDim.x) {
        unsigned long long index = block * blockDim.x + threadIdx.x;
        unsigned long long smallerCounter = 0;
        unsigned long long greaterCounter = 0;
        unsigned long long pivotCounter = 0;
        for (unsigned long long i = block * blockDim.x * elementsThread + threadIdx.x; i < partitionSize && i < (block + 1) * blockDim.x * elementsThread;
             i += blockDim.x) {
            if (partitionData[i] < *pivot) {
                newPartitionData[bSmaller[block] + tSmaller[index] + smallerCounter++] = partitionData[i];
            } else if (partitionData[i] > *pivot) {
                newPartitionData[*greaterOffset + bGreater[block] + tGreater[index] + greaterCounter++] = partitionData[i];
            } else {
                newPartitionData[*pivotOffset + bPivot[block] + tPivot[index] + pivotCounter++] = partitionData[i];
            }
        }
    }
}

#endif