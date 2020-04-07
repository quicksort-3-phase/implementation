#ifndef QUICKSORT_H
#define QUICKSORT_H

/**
 * Algorithm due to Daniel Cederman & Philippas Tsigas
 * Based on http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.156.9622&rep=rep1&type=pdf
 * Implemented by Alexander Fischer
 */

#include <algorithm>
#include <cmath>
#include <iostream>

#include "helper.h"

template <typename T>
__device__ void _quicksort_medianOfFive(T *data0, T *data1, T *data2, T *data3, T *data4) {
    sortSwap(data0, data1);
    sortSwap(data3, data4);
    sortSwap(data0, data3);
    sortSwap(data1, data4);
    sortSwap(data2, data3);
    sortSwap(data1, data2);
    sortSwap(data2, data3);
}

template <typename T>
__device__ void quicksort_medianOfFiveInRow(T *data) {
    sortSwap(&data[0], &data[1]);
    sortSwap(&data[3], &data[4]);
    sortSwap(&data[0], &data[3]);
    sortSwap(&data[1], &data[4]);
    sortSwap(&data[2], &data[3]);
    sortSwap(&data[1], &data[2]);
    sortSwap(&data[2], &data[3]);
}

template <typename T>
__device__ T quicksort_medianOfFive(T *data, uint32_t size) {
    _quicksort_medianOfFive(&data[0], &data[size / 4], &data[size / 2], &data[(3 * size) / 4], &data[size - 1]);
    return data[size / 2];
}

template <typename T>
__device__ T quicksort_medianOfFiveMedianOfFive(T *data, uint32_t size) {
    quicksort_medianOfFiveInRow(&data[0]);
    quicksort_medianOfFiveInRow(&data[size / 4]);
    quicksort_medianOfFiveInRow(&data[size / 2]);
    quicksort_medianOfFiveInRow(&data[(3 * size) / 4]);
    quicksort_medianOfFiveInRow(&data[size - 6]);
    _quicksort_medianOfFive(&data[2], &data[size / 4 + 2], &data[size / 2 + 2], &data[(3 * size) / 4 + 2], &data[size - 4]);
    return data[size / 2 + 2];
}

template <typename T>
__device__ void _quicksort_medianOfThree(T *data0, T *data1, T *data2) {
    sortSwap(data0, data1);
    sortSwap(data1, data2);
    sortSwap(data0, data1);
}

template <typename T>
__device__ void quicksort_medianOfThreeInRow(T *data) {
    sortSwap(&data[0], &data[1]);
    sortSwap(&data[1], &data[2]);
    sortSwap(&data[0], &data[1]);
}

template <typename T>
__device__ T quicksort_medianOfThree(T *data, uint32_t size) {
    T first = data[0];
    T second = data[size / 2];
    T third = data[size - 1];
    if ((first > second) != (first > third)) {
        return first;
    } else if ((second > first) != (second > third)) {
        return second;
    } else {
        return third;
    }
}

template <typename T>
__device__ T quicksort_medianOfThreeMedianOfThree(T *data, uint32_t size) {
    quicksort_medianOfThreeInRow(&data[0]);
    quicksort_medianOfThreeInRow(&data[size / 2]);
    quicksort_medianOfThreeInRow(&data[size - 4]);
    _quicksort_medianOfThree(&data[1], &data[size / 2 + 1], &data[size - 3]);
    return data[size / 2 + 1];
}

template <typename T>
__device__ T quicksort_medianOfThreeMedianOfFive(T *data, uint32_t size) {
    quicksort_medianOfFiveInRow(&data[0]);
    quicksort_medianOfFiveInRow(&data[size / 2]);
    quicksort_medianOfFiveInRow(&data[size - 6]);
    _quicksort_medianOfThree(&data[2], &data[size / 2 + 2], &data[size - 4]);
    return data[size / 2 + 2];
}

template <typename T>
__global__ void quicksort_allSingleThreaded(T *currentData, T *newData, uint32_t singleThreadCount, uint32_t *currentPartitions, uint32_t *currentPartitionSizes,
                                            uint32_t *pivotOffsets, uint32_t *greaterOffsets) {
    for (uint32_t partition = blockIdx.x * blockDim.x + threadIdx.x; partition < singleThreadCount; partition += gridDim.x * blockDim.x) {
        uint32_t partitionSize = currentPartitionSizes[partition + 1];
        uint32_t left = currentPartitions[partition];
        T *partitionData = &currentData[left];
        T *newPartitionData = &newData[left];
        uint32_t smallerCounter = 0;
        uint32_t greaterCounter = 0;
        uint32_t pivotCounter = 0;
        T pivot = quicksort_medianOfThree(&currentData[left], partitionSize);

        // Count/Move elements
        for (uint32_t i = 0; i < partitionSize; i++) {
            if (partitionData[i] < pivot) {
                newPartitionData[smallerCounter++] = partitionData[i];
            } else if (partitionData[i] > pivot) {
                newPartitionData[partitionSize - 1 - greaterCounter++] = partitionData[i];
            }
        }
        for (uint32_t i = 0; i < partitionSize; i++) {
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
__global__ void quicksort_pivot(T *data, T *pivots, uint32_t *multiThreadPartitions, uint32_t *multiThreadFirstPartitionIndex, uint32_t multiThreadPartitionCount,
                                uint32_t *partitions, uint32_t *partitionSizes, uint32_t pivot_type = 0) {
    for (uint32_t multiThreadPartition = blockIdx.x * blockDim.x + threadIdx.x; multiThreadPartition < multiThreadPartitionCount;
         multiThreadPartition += gridDim.x * blockDim.x) {
        uint32_t partition = multiThreadPartitions[multiThreadFirstPartitionIndex[multiThreadPartition]];
        uint32_t partitionSize = partitionSizes[partition + 1];
        switch (pivot_type) {
            case 0:
                if (partitionSize > 25) {
                    pivots[multiThreadPartition] = quicksort_medianOfFiveMedianOfFive(&data[partitions[partition]], partitionSize);
                } else {
                    pivots[multiThreadPartition] = quicksort_medianOfThree(&data[partitions[partition]], partitionSize);
                }
                break;
            case 1:
                pivots[multiThreadPartition] = quicksort_medianOfThree(&data[partitions[partition]], partitionSize);
                break;
            case 2:
                pivots[multiThreadPartition] = quicksort_medianOfFive(&data[partitions[partition]], partitionSize);
                break;
            case 3:
                pivots[multiThreadPartition] = quicksort_medianOfThreeMedianOfThree(&data[partitions[partition]], partitionSize);
                break;
            case 4:
                pivots[multiThreadPartition] = quicksort_medianOfThreeMedianOfFive(&data[partitions[partition]], partitionSize);
                break;
            case 5:
                pivots[multiThreadPartition] = quicksort_medianOfFiveMedianOfFive(&data[partitions[partition]], partitionSize);
                break;
        }
    }
}

template <typename T>
__global__ void quicksort_singleThreaded(T *currentData, T *newData, uint32_t *pivotOffsets, uint32_t *greaterOffsets, uint32_t *singleThreadPartitions,
                                         uint32_t singleThreadCount, uint32_t *currentPartitions, uint32_t *currentPartitionSizes) {
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < singleThreadCount; index += gridDim.x * blockDim.x) {
        uint32_t partition = singleThreadPartitions[index];
        uint32_t left = currentPartitions[partition];
        T *partitionData = &currentData[left];
        T *newPartitionData = &newData[left];
        uint32_t partitionSize = currentPartitionSizes[partition + 1];
        uint32_t smallerCounter = 0;
        uint32_t greaterCounter = 0;
        uint32_t pivotCounter = 0;
        T pivot = quicksort_medianOfThree(&currentData[left], partitionSize);

        // Count/Move elements
        for (uint32_t i = 0; i < partitionSize; i++) {
            if (partitionData[i] < pivot) {
                newPartitionData[smallerCounter++] = partitionData[i];
            } else if (partitionData[i] > pivot) {
                newPartitionData[partitionSize - 1 - greaterCounter++] = partitionData[i];
            }
        }
        for (uint32_t i = 0; i < partitionSize; i++) {
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
__global__ void quicksort_multiThreaded1(T *currentData, T *pivots, uint32_t *multiThreadPartitions, uint32_t *multiThreadPartitionMembers,
                                         uint32_t *partitionMultiThreadPartitions, uint32_t multiThreadCount, uint32_t *currentPartitions,
                                         uint32_t *currentPartitionSizes, uint32_t *tSmaller, uint32_t *tGreater, uint32_t *tPivot, uint32_t elementsThread) {
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < multiThreadCount; index += gridDim.x * blockDim.x) {
        uint32_t partition = multiThreadPartitions[index];
        uint32_t partitionMember = multiThreadPartitionMembers[index];
        T *partitionData = &currentData[currentPartitions[partition]];
        uint32_t partitionSize = currentPartitionSizes[partition + 1];
        uint32_t memberSize = ceil((double)partitionSize / elementsThread);
        T pivot = pivots[partitionMultiThreadPartitions[partition]];

        uint32_t smallerElements = 0;
        uint32_t greaterElements = 0;
        uint32_t pivotElements = 0;
        for (uint32_t i = partitionMember; i < partitionSize; i += memberSize) {
#if false
            T element = partitionData[i];
            smallerElements += element < pivot;
            greaterElements += element > pivot;
            pivotElements += element == pivot;
#else
            if (partitionData[i] < pivot) {
                smallerElements++;
            } else if (partitionData[i] > pivot) {
                greaterElements++;
            } else {
                pivotElements++;
            }
#endif
        }
        tSmaller[index] = smallerElements;
        tGreater[index] = greaterElements;
        tPivot[index] = pivotElements;
    }
}

#define MAX_SCAN_BLOCKS 1024
#define SCAN_THREADS 128
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
__global__ void scanBlocks(uint32_t *multiThreadPartitions, uint32_t *multiThreadBlockOffset, uint32_t *multiThreadFirstPartitionIndex,
                           uint32_t multiThreadPartitionCount, uint32_t *currentPartitionSizes, uint32_t *tSmaller, uint32_t *bSmaller, uint32_t *tGreater,
                           uint32_t *bGreater, uint32_t *tPivot, uint32_t *bPivot, uint32_t elementsThread) {
    extern __shared__ uint32_t temp[];
    uint32_t index = threadIdx.x;
    for (uint32_t multiThreadPartition = 0; multiThreadPartition < multiThreadPartitionCount; multiThreadPartition++) {
        uint32_t firstIndex = multiThreadFirstPartitionIndex[multiThreadPartition];
        uint32_t partition = multiThreadPartitions[firstIndex];
        uint32_t size = ceil((double)currentPartitionSizes[partition + 1] / elementsThread);
        uint32_t bOffset = multiThreadBlockOffset[multiThreadPartition];
        uint32_t blocks = ceil((double)size / SCAN_THREADS);

        for (uint32_t block = blockIdx.x; block < 3 * blocks; block += gridDim.x) {
            uint32_t *dataPtr;
            uint32_t *blockPtr;
            switch (block / blocks) {
                case 0:
                    dataPtr = &tSmaller[firstIndex];
                    blockPtr = &bSmaller[bOffset];
                    break;
                case 1:
                    dataPtr = &tGreater[firstIndex];
                    blockPtr = &bGreater[bOffset];
                    break;
                case 2:
                    dataPtr = &tPivot[firstIndex];
                    blockPtr = &bPivot[bOffset];
                    break;
            }

            uint32_t blockOffset = (block % blocks) * SCAN_THREADS;
            uint32_t offset = 1;

            // load input into shared memory
            uint32_t ai = index;
            uint32_t bi = index + (SCAN_THREADS / 2);
            uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
            if (blockOffset + ai < size) {
                temp[ai + bankOffsetA] = dataPtr[blockOffset + ai];
            } else {
                temp[ai + bankOffsetA] = 0;
            }
            if (blockOffset + bi < size) {
                temp[bi + bankOffsetB] = dataPtr[blockOffset + bi];
            } else {
                temp[bi + bankOffsetB] = 0;
            }

            // build sum in place up the tree
            for (uint32_t d = SCAN_THREADS >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (index < d) {
                    uint32_t ai = offset * (2 * index + 1) - 1;
                    uint32_t bi = offset * (2 * index + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            // clear the last element and store block sum
            if (index == 0) {
                blockPtr[block % blocks] = temp[SCAN_THREADS - 1];
                temp[SCAN_THREADS - 1] = 0;
            }

            // traverse down tree & build scan
            for (uint32_t d = 1; d < SCAN_THREADS; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (index < d) {
                    uint32_t ai = offset * (2 * index + 1) - 1;
                    uint32_t bi = offset * (2 * index + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    float t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            // write results to device memory
            if (blockOffset + ai < size) {
                dataPtr[blockOffset + ai] = temp[ai + bankOffsetA];
            }
            if (blockOffset + bi < size) {
                dataPtr[blockOffset + bi] = temp[bi + bankOffsetB];
            }
        }
    }
}

__global__ void quicksort_multiThreaded2_parallel(uint32_t *pivotOffsets, uint32_t *greaterOffsets, uint32_t *multiThreadPartitions, uint32_t *multiThreadBlockOffset,
                                                  uint32_t *multiThreadFirstPartitionIndex, uint32_t multiThreadPartitionCount, uint32_t *currentPartitionSizes,
                                                  uint32_t *bSmaller, uint32_t *bGreater, uint32_t *bPivot, uint32_t elementsThread) {
    for (uint32_t multiThreadPartition = blockIdx.x * blockDim.x + threadIdx.x; multiThreadPartition < multiThreadPartitionCount;
         multiThreadPartition += gridDim.x * blockDim.x) {
        uint32_t firstIndex = multiThreadFirstPartitionIndex[multiThreadPartition];
        uint32_t partition = multiThreadPartitions[firstIndex];
        uint32_t memberSize = ceil((double)currentPartitionSizes[partition + 1] / elementsThread);
        uint32_t blockOffset = multiThreadBlockOffset[multiThreadPartition];

        uint32_t blocks = ceil((double)memberSize / SCAN_THREADS);
        uint32_t lastSmaller = bSmaller[blockOffset];
        uint32_t lastPivot = bPivot[blockOffset];

        if (blocks > 1) {
            uint32_t lastGreater = bGreater[blockOffset];
            uint32_t tempSmaller;
            uint32_t tempGreater;
            uint32_t tempPivot;
            bSmaller[blockOffset] = 0;
            bGreater[blockOffset] = 0;
            bPivot[blockOffset] = 0;
            for (uint32_t block = blockOffset + 1; block < blockOffset + blocks; block++) {
                tempSmaller = bSmaller[block];
                tempGreater = bGreater[block];
                tempPivot = bPivot[block];
                bSmaller[block] = lastSmaller + bSmaller[block - 1];
                bGreater[block] = lastGreater + bGreater[block - 1];
                bPivot[block] = lastPivot + bPivot[block - 1];
                lastSmaller = tempSmaller;
                lastGreater = tempGreater;
                lastPivot = tempPivot;
            }

            uint32_t pOffset = bSmaller[blockOffset + blocks - 1] + lastSmaller;
            pivotOffsets[partition] = pOffset;
            greaterOffsets[partition] = pOffset + bPivot[blockOffset + blocks - 1] + lastPivot;
        } else {
            bSmaller[blockOffset] = 0;
            bGreater[blockOffset] = 0;
            bPivot[blockOffset] = 0;
            pivotOffsets[partition] = lastSmaller;
            greaterOffsets[partition] = lastSmaller + lastPivot;
        }
        assert(pivotOffsets[partition] < greaterOffsets[partition]);
    }
}

__global__ void quicksort_multiThreaded2_sequential(uint32_t *pivotOffsets, uint32_t *greaterOffsets, uint32_t *multiThreadPartitions,
                                                    uint32_t *multiThreadFirstPartitionIndex, uint32_t multiThreadPartitionCount, uint32_t *currentPartitionSizes,
                                                    uint32_t *tSmaller, uint32_t *tGreater, uint32_t *tPivot, uint32_t elementsThread) {
    for (uint32_t multiThreadPartition = blockIdx.x * blockDim.x + threadIdx.x; multiThreadPartition < multiThreadPartitionCount;
         multiThreadPartition += gridDim.x * blockDim.x) {
        uint32_t firstIndex = multiThreadFirstPartitionIndex[multiThreadPartition];
        uint32_t partition = multiThreadPartitions[firstIndex];
        uint32_t memberSize = ceil((double)currentPartitionSizes[partition + 1] / elementsThread);

        uint32_t lastSmaller = tSmaller[firstIndex];
        uint32_t lastGreater = tGreater[firstIndex];
        uint32_t lastPivot = tPivot[firstIndex];
        uint32_t tempSmaller;
        uint32_t tempGreater;
        uint32_t tempPivot;
        tSmaller[firstIndex] = 0;
        tGreater[firstIndex] = 0;
        tPivot[firstIndex] = 0;
        for (uint32_t partitionMember = firstIndex + 1; partitionMember < firstIndex + memberSize; partitionMember++) {
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

        uint32_t pOffset = tSmaller[firstIndex + memberSize - 1] + lastSmaller;
        pivotOffsets[partition] = pOffset;
        greaterOffsets[partition] = pOffset + tPivot[firstIndex + memberSize - 1] + lastPivot;
    }
}

template <typename T>
__global__ void quicksort_multiThreaded3(T *currentData, T *newData, T *pivots, uint32_t *pivotOffsets, uint32_t *greaterOffsets, uint32_t *multiThreadPartitions,
                                         uint32_t *multiThreadPartitionMembers, uint32_t *multiThreadBlockOffset, uint32_t *partitionMultiThreadPartitions,
                                         uint32_t multiThreadCount, uint32_t *currentPartitions, uint32_t *currentPartitionSizes, uint32_t *tSmaller, uint32_t *tGreater,
                                         uint32_t *tPivot, uint32_t *bSmaller, uint32_t *bGreater, uint32_t *bPivot, bool useParallelScan, uint32_t elementsThread) {
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < multiThreadCount; index += gridDim.x * blockDim.x) {
        uint32_t partition = multiThreadPartitions[index];
        uint32_t partitionMember = multiThreadPartitionMembers[index];
        T *partitionData = &currentData[currentPartitions[partition]];
        T *newPartitionData = &newData[currentPartitions[partition]];
        uint32_t partitionSize = currentPartitionSizes[partition + 1];
        uint32_t memberSize = ceil((double)partitionSize / elementsThread);
        uint32_t multiThreadPartition = partitionMultiThreadPartitions[partition];
        uint32_t blockOffset = multiThreadBlockOffset[multiThreadPartition] + partitionMember / SCAN_THREADS;
        T pivot = pivots[multiThreadPartition];

        uint32_t smallerCounter = 0;
        uint32_t greaterCounter = 0;
        uint32_t pivotCounter = 0;
        uint32_t smallerTotalOffset = tSmaller[index] + (useParallelScan ? bSmaller[blockOffset] : 0);
        uint32_t greaterTotalOffset = greaterOffsets[partition] + tGreater[index] + (useParallelScan ? bGreater[blockOffset] : 0);
        uint32_t pivotTotalOffset = pivotOffsets[partition] + tPivot[index] + (useParallelScan ? bPivot[blockOffset] : 0);
        for (uint32_t i = partitionMember; i < partitionSize; i += memberSize) {
#if false
            bool isSmaller = partitionData[i] < pivot;
            bool isGreater = partitionData[i] > pivot;
            newPartitionData[isSmaller * (smallerTotalOffset + smallerCounter) + isGreater * (greaterTotalOffset + greaterCounter) +
                             !(isSmaller || isGreater) * (pivotTotalOffset + pivotCounter)] = partitionData[i];
            smallerCounter += isSmaller;
            greaterCounter += isGreater;
            pivotCounter += !(isSmaller || isGreater);
#else
            if (partitionData[i] < pivot) {
                newPartitionData[smallerTotalOffset + smallerCounter++] = partitionData[i];
            } else if (partitionData[i] > pivot) {
                newPartitionData[greaterTotalOffset + greaterCounter++] = partitionData[i];
            } else {
                newPartitionData[pivotTotalOffset + pivotCounter++] = partitionData[i];
            }
#endif
        }
    }
}

template <typename T>
__global__ void quicksort_createPartitions(T *currentData, T *newData, uint32_t *pivotOffsets, uint32_t *greaterOffsets, uint32_t *currentPartitions,
                                           uint32_t *newPartitions, uint32_t *currentPartitionSizes, uint32_t *newPartitionSizes, uint32_t threshold,
                                           bool isNewDataFinalData, uint32_t *finishedPartitions, uint32_t *finishedPartitionSizes) {
    uint32_t partitionCount = currentPartitionSizes[0];

    if (threadIdx.x == 0) {
        currentPartitionSizes[0] = 0;
        newPartitionSizes[0] = 0;
        if (threshold > 1) {
            finishedPartitionSizes[0] = 0;
        }
    }
    __syncthreads();

    for (uint32_t partition = threadIdx.x; partition < partitionCount; partition += blockDim.x) {
        if (pivotOffsets[partition] <= threshold) {
            if (!isNewDataFinalData) {
                for (uint32_t i = currentPartitions[partition]; i < currentPartitions[partition] + pivotOffsets[partition]; i++) {
                    currentData[i] = newData[i];
                }
            }
            if (threshold > 1 && pivotOffsets[partition] > 1) {
                uint32_t index = atomicAdd(&finishedPartitionSizes[0], 1);
                finishedPartitions[index] = currentPartitions[partition];
                finishedPartitionSizes[index + 1] = pivotOffsets[partition];
            }
        } else {
            uint32_t index = atomicAdd(&newPartitionSizes[0], 1);
            newPartitions[index] = currentPartitions[partition];
            newPartitionSizes[index + 1] = pivotOffsets[partition];
        }
        if (currentPartitionSizes[partition + 1] - greaterOffsets[partition] <= threshold) {
            if (!isNewDataFinalData) {
                for (uint32_t i = currentPartitions[partition] + greaterOffsets[partition]; i < currentPartitions[partition] + currentPartitionSizes[partition + 1];
                     i++) {
                    currentData[i] = newData[i];
                }
            }
            if (threshold > 1 && currentPartitionSizes[partition + 1] - greaterOffsets[partition] > 1) {
                uint32_t index = atomicAdd(&finishedPartitionSizes[0], 1);
                finishedPartitions[index] = currentPartitions[partition] + greaterOffsets[partition];
                finishedPartitionSizes[index + 1] = currentPartitionSizes[partition + 1] - greaterOffsets[partition];
            }
        } else {
            uint32_t index = atomicAdd(&newPartitionSizes[0], 1);
            newPartitions[index] = currentPartitions[partition] + greaterOffsets[partition];
            newPartitionSizes[index + 1] = currentPartitionSizes[partition + 1] - greaterOffsets[partition];
        }

        if (!isNewDataFinalData) {
            for (uint32_t i = currentPartitions[partition] + pivotOffsets[partition]; i < currentPartitions[partition] + greaterOffsets[partition]; i++) {
                currentData[i] = newData[i];
            }
        }
    }
}

template <typename T>
__global__ void quicksort_sortingNetwork16(T *data, uint32_t *finishedPartitions, uint32_t *finishedPartitionSizes) {
    uint32_t partitionCount = finishedPartitionSizes[0];
    for (uint32_t partition = blockIdx.x * blockDim.x + threadIdx.x; partition < partitionCount; partition += gridDim.x * blockDim.x) {
        sortingNetwork16(&data[finishedPartitions[partition]], finishedPartitionSizes[partition + 1]);
    }
}

template <typename T>
__global__ void quicksort_sortingNetwork24(T *data, uint32_t *finishedPartitions, uint32_t *finishedPartitionSizes) {
    uint32_t partitionCount = finishedPartitionSizes[0];
    for (uint32_t partition = blockIdx.x * blockDim.x + threadIdx.x; partition < partitionCount; partition += gridDim.x * blockDim.x) {
        sortingNetwork24(&data[finishedPartitions[partition]], finishedPartitionSizes[partition + 1]);
    }
}

template <typename T>
__global__ void quicksort_sortingNetwork32(T *data, uint32_t *finishedPartitions, uint32_t *finishedPartitionSizes) {
    uint32_t partitionCount = finishedPartitionSizes[0];
    for (uint32_t partition = blockIdx.x * blockDim.x + threadIdx.x; partition < partitionCount; partition += gridDim.x * blockDim.x) {
        sortingNetwork32(&data[finishedPartitions[partition]], finishedPartitionSizes[partition + 1]);
    }
}

template <typename T>
int quicksort(uint32_t size, T *h_data, double &startTime, double &endTime, size_t &endfreeMem, size_t &totalMem, uint32_t threshold, uint32_t maxBlocks,
              uint32_t qs2Threads, uint32_t elementsThread, uint32_t scanThreshold, uint32_t pivot_type) {
    if (size < 2) {
        return 0;
    }
    if (maxBlocks == 0) {
        maxBlocks = 32;
    }
    if (qs2Threads == 0) {
        if (size < 20000) {
            qs2Threads = 768;
        } else if (size < 250000) {
            qs2Threads = 256;
        } else if (size < 3000000) {
            qs2Threads = 1024;
        } else if (size < 8000000) {
            qs2Threads = 256;
        } else {
            qs2Threads = 128;
        }
    }
    if (elementsThread == 0) {
        if (size < 250) {
            elementsThread = 512;
        } else if (size < 400000) {
            elementsThread = 64;
        } else if (size < 750000) {
            elementsThread = 128;
        } else if (size < 1500000) {
            elementsThread = 256;
        } else if (size < 3000000) {
            elementsThread = 512;
        } else if (size < 6000000) {
            elementsThread = 1024;
        } else {
            elementsThread = 64;
        }
    }
    if (scanThreshold == 0) {
        if (size < 10000) {
            scanThreshold = 2048;
        } else if (size < 80000) {
            scanThreshold = 128;
        } else if (size < 200000) {
            scanThreshold = 256;
        } else if (size < 5000000) {
            scanThreshold = 512;
        } else if (size < 20000000) {
            scanThreshold = 2048;
        } else {
            scanThreshold = 4096;
        }
    }

    T *d_data1, *d_data2, *d_currentData, *d_newData, *d_pivot;
    // Partition handling, all partitionSizes-arrays contain size in first element
    uint32_t *d_pivotOffsets, *d_greaterOffsets, *d_partitions1, *d_partitions2, *d_currentPartitions, *d_newPartitions, *d_partitionSizes1, *d_partitionSizes2,
        *d_currentPartitionSizes, *d_newPartitionSizes, *h_partitionSizes;
    // Arrays for numbers of elements greater/smaller than the pivot (threads)
    uint32_t *d_tSmaller, *d_tGreater, *d_tPivot;
    // Arrays for cumulative sums (blocks)
    uint32_t *d_bSmaller, *d_bGreater, *d_bPivot;
    // Arrays to handle bulk kernel
    uint32_t *d_singleThreadPartitions, *h_singleThreadPartitions;                  // Maps singleThread index to partition index
    uint32_t *d_multiThreadPartitions, *h_multiThreadPartitions;                    // Maps multiThread index to partition index
    uint32_t *d_multiThreadPartitionMembers, *h_multiThreadPartitionMembers;        // Maps multiThread index to relative thread index inside its partition
    uint32_t *d_multiThreadBlockOffset, *h_multiThreadBlockOffset;                  // Maps multiThread partition to block offset (used for bigger prefix sums)
    uint32_t *d_multiThreadFirstPartitionIndex, *h_multiThreadFirstPartitionIndex;  // Maps multiThread partition to its first multiThread index
    uint32_t *d_partitionMultiThreadPartitions, *h_partitionMultiThreadPartitions;  // Maps partition index to multiThread partition
    // Storage of finished partitions (threshold) to sort differently
    uint32_t *d_finishedPartitions, *d_finishedPartitionSizes;

    int greatestPriority, leastPriority;
    gpuErrchk(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    cudaStream_t mainStream;
    cudaStream_t singleThreadStream;
    cudaStream_t multiThreadStream;
    cudaStream_t backgroundStream;
    gpuErrchk(cudaStreamCreateWithPriority(&mainStream, 0, greatestPriority));
    gpuErrchk(cudaStreamCreateWithPriority(&singleThreadStream, 0, greatestPriority));
    gpuErrchk(cudaStreamCreateWithPriority(&multiThreadStream, 0, greatestPriority));
    gpuErrchk(cudaStreamCreateWithPriority(&backgroundStream, 0, leastPriority));

    uint32_t zero = 0;
    uint32_t maxThresholdSize = std::ceil((double)size / (threshold + 1));
    uint32_t maxMultiThreadSize = 2 * std::ceil((double)size / elementsThread);
    uint32_t maxMultiThreadPartitionSize = std::ceil((double)size / std::max(elementsThread, threshold));
    uint32_t maxScanBlockSize = std::max((uint32_t)std::ceil(2 * (double)size / SCAN_THREADS), maxMultiThreadSize);

    gpuErrchk(cudaMallocHost(&h_partitionSizes, (maxThresholdSize + 1) * sizeof(uint32_t)));
    gpuErrchk(cudaMallocHost(&h_singleThreadPartitions, maxThresholdSize * sizeof(uint32_t)));
    gpuErrchk(cudaMallocHost(&h_multiThreadPartitions, maxMultiThreadSize * sizeof(uint32_t)));
    gpuErrchk(cudaMallocHost(&h_multiThreadPartitionMembers, maxMultiThreadSize * sizeof(uint32_t)));
    gpuErrchk(cudaMallocHost(&h_multiThreadBlockOffset, maxMultiThreadPartitionSize * sizeof(uint32_t)));
    gpuErrchk(cudaMallocHost(&h_multiThreadFirstPartitionIndex, maxMultiThreadPartitionSize * sizeof(uint32_t)));
    gpuErrchk(cudaMallocHost(&h_partitionMultiThreadPartitions, maxThresholdSize * sizeof(uint32_t)));

    gpuErrchk(cudaMalloc(&d_data1, size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_data2, size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_pivot, maxMultiThreadPartitionSize * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_pivotOffsets, maxThresholdSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_greaterOffsets, maxThresholdSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_partitions1, maxThresholdSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_partitions2, maxThresholdSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_partitionSizes1, (maxThresholdSize + 1) * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_partitionSizes2, (maxThresholdSize + 1) * sizeof(uint32_t)));

    gpuErrchk(cudaMalloc(&d_tSmaller, maxMultiThreadSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_tGreater, maxMultiThreadSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_tPivot, maxMultiThreadSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_bSmaller, maxScanBlockSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_bGreater, maxScanBlockSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_bPivot, maxScanBlockSize * sizeof(uint32_t)));

    gpuErrchk(cudaMalloc(&d_singleThreadPartitions, maxThresholdSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_multiThreadPartitions, maxMultiThreadSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_multiThreadPartitionMembers, maxMultiThreadSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_multiThreadBlockOffset, maxMultiThreadPartitionSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_multiThreadFirstPartitionIndex, maxMultiThreadPartitionSize * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_partitionMultiThreadPartitions, maxThresholdSize * sizeof(uint32_t)));

    if (threshold > 1) {
        gpuErrchk(cudaMalloc(&d_finishedPartitions, maxThresholdSize * sizeof(uint32_t)));
        gpuErrchk(cudaMalloc(&d_finishedPartitionSizes, (maxThresholdSize + 1) * sizeof(uint32_t)));
    }

    gpuErrchk(cudaMemcpy(d_data1, h_data, size * sizeof(T), cudaMemcpyHostToDevice));

    cudaMemGetInfo(&endfreeMem, &totalMem);

    gpuErrchk(cudaDeviceSynchronize());
    startTime = clock();

    d_currentData = d_data1;
    d_newData = d_data2;
    d_currentPartitions = d_partitions1;
    d_newPartitions = d_partitions2;
    d_currentPartitionSizes = d_partitionSizes1;
    d_newPartitionSizes = d_partitionSizes2;

    gpuErrchk(cudaMemcpyAsync(d_currentPartitions, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, mainStream));
    h_partitionSizes[0] = 1;
    h_partitionSizes[1] = size;
    gpuErrchk(cudaMemcpyAsync(d_currentPartitionSizes, h_partitionSizes, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, mainStream));
    gpuErrchk(cudaStreamSynchronize(mainStream));

    do {
        // Put partitions into single/multi thread arrays
        uint32_t threadCountOfBiggestPartition = 0;
        uint32_t partitionCount = h_partitionSizes[0];
        uint32_t singleThreadCount = 0;
        uint32_t multiThreadCount = 0;
        uint32_t multiThreadPartitionCount = 0;
        uint32_t lastSize = 0;
        for (uint32_t partition = 0; partition < partitionCount; partition++) {
            uint32_t partitionSize = h_partitionSizes[partition + 1];
            if (partitionSize > elementsThread) {
                uint32_t threadsNeeded = std::ceil((double)partitionSize / elementsThread);
                if (threadsNeeded > threadCountOfBiggestPartition) {
                    threadCountOfBiggestPartition = threadsNeeded;
                }

                // Store blockOffset for scanning and first index of each multithread partition
                uint32_t tempSize = std::ceil((double)partitionSize / SCAN_THREADS);
                if (multiThreadPartitionCount < 1) {
                    h_multiThreadBlockOffset[0] = 0;
                } else {
                    h_multiThreadBlockOffset[multiThreadPartitionCount] = h_multiThreadBlockOffset[multiThreadPartitionCount - 1] + lastSize;
                }
                lastSize = tempSize;
                h_multiThreadFirstPartitionIndex[multiThreadPartitionCount] = multiThreadCount;
                h_partitionMultiThreadPartitions[partition] = multiThreadPartitionCount;
                multiThreadPartitionCount++;

                for (uint32_t partitionMember = 0; partitionMember < std::ceil((double)partitionSize / elementsThread); partitionMember++) {
                    h_multiThreadPartitions[multiThreadCount] = partition;
                    h_multiThreadPartitionMembers[multiThreadCount] = partitionMember;
                    multiThreadCount++;
                }
            } else {
                h_singleThreadPartitions[singleThreadCount++] = partition;
            }
        }

        if (multiThreadPartitionCount == 0) {
            // Sort all single-threaded
            uint32_t blocks = std::min((uint32_t)std::ceil((double)singleThreadCount / qs2Threads), maxBlocks);
            uint32_t threads = std::min(singleThreadCount, qs2Threads);
            quicksort_allSingleThreaded<<<blocks, threads, 0, mainStream>>>(d_currentData, d_newData, singleThreadCount, d_currentPartitions, d_currentPartitionSizes,
                                                                            d_pivotOffsets, d_greaterOffsets);
        } else {
            // Sort single thread partitions
            if (singleThreadCount > 0) {
                gpuErrchk(cudaMemcpyAsync(d_singleThreadPartitions, h_singleThreadPartitions, singleThreadCount * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                          singleThreadStream));
                uint32_t blocks = std::min((uint32_t)std::ceil((double)singleThreadCount / qs2Threads), maxBlocks);
                uint32_t threads = std::min(singleThreadCount, qs2Threads);
                quicksort_singleThreaded<<<blocks, threads, 0, singleThreadStream>>>(d_currentData, d_newData, d_pivotOffsets, d_greaterOffsets, d_singleThreadPartitions,
                                                                                     singleThreadCount, d_currentPartitions, d_currentPartitionSizes);
            }

            // Sort multi thread partitions
            if (multiThreadCount > 0) {
                gpuErrchk(
                    cudaMemcpyAsync(d_multiThreadPartitions, h_multiThreadPartitions, multiThreadCount * sizeof(uint32_t), cudaMemcpyHostToDevice, multiThreadStream));
                gpuErrchk(cudaMemcpyAsync(d_multiThreadPartitionMembers, h_multiThreadPartitionMembers, multiThreadCount * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                          multiThreadStream));
                gpuErrchk(cudaMemcpyAsync(d_multiThreadBlockOffset, h_multiThreadBlockOffset, multiThreadPartitionCount * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                          multiThreadStream));
                gpuErrchk(cudaMemcpyAsync(d_multiThreadFirstPartitionIndex, h_multiThreadFirstPartitionIndex, multiThreadPartitionCount * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice, multiThreadStream));
                gpuErrchk(cudaMemcpyAsync(d_partitionMultiThreadPartitions, h_partitionMultiThreadPartitions, partitionCount * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                          multiThreadStream));

                uint32_t pBlocks = std::min((uint32_t)std::ceil((double)multiThreadPartitionCount / qs2Threads), maxBlocks);
                uint32_t pThreads = std::min(partitionCount, qs2Threads);
                quicksort_pivot<<<pBlocks, pThreads, 0, multiThreadStream>>>(d_currentData, d_pivot, d_multiThreadPartitions, d_multiThreadFirstPartitionIndex,
                                                                             multiThreadPartitionCount, d_currentPartitions, d_currentPartitionSizes, pivot_type);

                uint32_t blocks = std::min((uint32_t)std::ceil((double)multiThreadCount / qs2Threads), maxBlocks);
                uint32_t threads = std::min(multiThreadCount, qs2Threads);
                uint32_t countBlocks = std::min((uint32_t)std::ceil((double)multiThreadPartitionCount / qs2Threads), maxBlocks);
                uint32_t countThreads = std::min(multiThreadPartitionCount, qs2Threads);

                quicksort_multiThreaded1<<<blocks, threads, 0, multiThreadStream>>>(d_currentData, d_pivot, d_multiThreadPartitions, d_multiThreadPartitionMembers,
                                                                                    d_partitionMultiThreadPartitions, multiThreadCount, d_currentPartitions,
                                                                                    d_currentPartitionSizes, d_tSmaller, d_tGreater, d_tPivot, elementsThread);

                bool useParallelScan = threadCountOfBiggestPartition > scanThreshold;
                if (useParallelScan) {
                    scanBlocks<<<std::min(3 * threadCountOfBiggestPartition, (uint32_t)MAX_SCAN_BLOCKS), SCAN_THREADS / 2, SCAN_THREADS * sizeof(uint32_t),
                                 multiThreadStream>>>(d_multiThreadPartitions, d_multiThreadBlockOffset, d_multiThreadFirstPartitionIndex, multiThreadPartitionCount,
                                                      d_currentPartitionSizes, d_tSmaller, d_bSmaller, d_tGreater, d_bGreater, d_tPivot, d_bPivot, elementsThread);
                    quicksort_multiThreaded2_parallel<<<countBlocks, countThreads, 0, multiThreadStream>>>(
                        d_pivotOffsets, d_greaterOffsets, d_multiThreadPartitions, d_multiThreadBlockOffset, d_multiThreadFirstPartitionIndex, multiThreadPartitionCount,
                        d_currentPartitionSizes, d_bSmaller, d_bGreater, d_bPivot, elementsThread);
                } else {
                    quicksort_multiThreaded2_sequential<<<countBlocks, countThreads, 0, multiThreadStream>>>(
                        d_pivotOffsets, d_greaterOffsets, d_multiThreadPartitions, d_multiThreadFirstPartitionIndex, multiThreadPartitionCount, d_currentPartitionSizes,
                        d_tSmaller, d_tGreater, d_tPivot, elementsThread);
                }
                quicksort_multiThreaded3<<<blocks, threads, 0, multiThreadStream>>>(
                    d_currentData, d_newData, d_pivot, d_pivotOffsets, d_greaterOffsets, d_multiThreadPartitions, d_multiThreadPartitionMembers, d_multiThreadBlockOffset,
                    d_partitionMultiThreadPartitions, multiThreadCount, d_currentPartitions, d_currentPartitionSizes, d_tSmaller, d_tGreater, d_tPivot, d_bSmaller,
                    d_bGreater, d_bPivot, useParallelScan, elementsThread);
            }
            gpuErrchk(cudaStreamSynchronize(singleThreadStream));
            gpuErrchk(cudaStreamSynchronize(multiThreadStream));
        }

        if (threshold > 1) {
            gpuErrchk(cudaStreamSynchronize(backgroundStream));
        }
        quicksort_createPartitions<<<1, std::min(partitionCount, (uint32_t)1024), 0, mainStream>>>(
            d_currentData, d_newData, d_pivotOffsets, d_greaterOffsets, d_currentPartitions, d_newPartitions, d_currentPartitionSizes, d_newPartitionSizes, threshold,
            d_newData == d_data2, d_finishedPartitions, d_finishedPartitionSizes);

        uint32_t finishedPartitionCount = 0;
        if (threshold > 1) {
            gpuErrchk(cudaMemcpyAsync(&finishedPartitionCount, d_finishedPartitionSizes, sizeof(uint32_t), cudaMemcpyDeviceToHost, mainStream));
        }
        gpuErrchk(cudaMemcpyAsync(h_partitionSizes, d_newPartitionSizes, std::min(2 * partitionCount + 1, maxThresholdSize) * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  mainStream));
        gpuErrchk(cudaStreamSynchronize(mainStream));
        if (threshold > 1 && finishedPartitionCount > 0) {
            // Capped to 384 to conform register limits of thread blocks for our Vector structs (of size 16 byte)
            uint32_t finishedBlocks = std::min((uint32_t)std::ceil((double)finishedPartitionCount / 384), maxBlocks);
            uint32_t finishedThreads = std::min(finishedPartitionCount, (uint32_t)384);
            if (threshold <= 16) {
                quicksort_sortingNetwork16<<<finishedBlocks, finishedThreads, 0, backgroundStream>>>(d_data2, d_finishedPartitions, d_finishedPartitionSizes);
            } else if (threshold <= 24) {
                quicksort_sortingNetwork24<<<finishedBlocks, finishedThreads, 0, backgroundStream>>>(d_data2, d_finishedPartitions, d_finishedPartitionSizes);
            } else if (threshold <= 32) {
                quicksort_sortingNetwork32<<<finishedBlocks, finishedThreads, 0, backgroundStream>>>(d_data2, d_finishedPartitions, d_finishedPartitionSizes);
            } else {
                std::cout << "Threshold not supported" << std::endl;
                return 1;
            }
        }

        // Swap pointers
        T *tempData = d_currentData;
        d_currentData = d_newData;
        d_newData = tempData;

        uint32_t *tempPartitions = d_currentPartitions;
        d_currentPartitions = d_newPartitions;
        d_newPartitions = tempPartitions;

        uint32_t *tempPartitionSizes = d_currentPartitionSizes;
        d_currentPartitionSizes = d_newPartitionSizes;
        d_newPartitionSizes = tempPartitionSizes;
    } while (h_partitionSizes[0] > 0);

    gpuErrchk(cudaDeviceSynchronize());
    endTime = clock();

    gpuErrchk(cudaMemcpy(h_data, d_data2, size * sizeof(T), cudaMemcpyDeviceToHost));

    cudaStreamDestroy(mainStream);
    cudaStreamDestroy(singleThreadStream);
    cudaStreamDestroy(multiThreadStream);
    cudaStreamDestroy(backgroundStream);
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
    cudaFree(d_multiThreadBlockOffset);
    cudaFree(d_multiThreadFirstPartitionIndex);
    cudaFree(d_partitionMultiThreadPartitions);
    if (threshold > 1) {
        cudaFree(d_finishedPartitions);
        cudaFree(d_finishedPartitionSizes);
    }
    cudaFreeHost(h_partitionSizes);
    cudaFreeHost(h_singleThreadPartitions);
    cudaFreeHost(h_multiThreadPartitions);
    cudaFreeHost(h_multiThreadPartitionMembers);
    cudaFreeHost(h_multiThreadBlockOffset);
    cudaFreeHost(h_multiThreadFirstPartitionIndex);
    cudaFreeHost(h_partitionMultiThreadPartitions);
    return 0;
}
#endif
