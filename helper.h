#ifndef HELPER_H
#define HELPER_H

#include <assert.h>
#include <limits>
#include "generators.h"

// Error handling in dynamic parallelism
#define cdpErrchk(ans) \
    { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) assert(0);
    }
}

// Error handling
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <typename T>
__device__ inline void swap(T *first, T *second) {
    T temp = *first;
    *first = *second;
    *second = temp;
}

template <typename T>
__device__ inline void sortSwap(T *first, T *second) {
    if (*first > *second) {
        T temp = *first;
        *first = *second;
        *second = temp;
    }
}

template <typename T>
__device__ void sortingNetwork16(T *data, uint32_t size) {
    T localData[16];
    for (uint32_t i = 0; i < size; i++) {
        localData[i] = data[i];
    }
    for (uint32_t i = size; i < 16; i++) {
        localData[i] = std::numeric_limits<T>::max();
    }

    sortSwap(&localData[0], &localData[1]);
    sortSwap(&localData[2], &localData[3]);
    sortSwap(&localData[4], &localData[5]);
    sortSwap(&localData[6], &localData[7]);
    sortSwap(&localData[8], &localData[9]);
    sortSwap(&localData[10], &localData[11]);
    sortSwap(&localData[12], &localData[13]);
    sortSwap(&localData[14], &localData[15]);
    sortSwap(&localData[0], &localData[2]);
    sortSwap(&localData[4], &localData[6]);
    sortSwap(&localData[8], &localData[10]);
    sortSwap(&localData[12], &localData[14]);
    sortSwap(&localData[1], &localData[3]);
    sortSwap(&localData[5], &localData[7]);
    sortSwap(&localData[9], &localData[11]);
    sortSwap(&localData[13], &localData[15]);
    sortSwap(&localData[0], &localData[4]);
    sortSwap(&localData[8], &localData[12]);
    sortSwap(&localData[1], &localData[5]);
    sortSwap(&localData[9], &localData[13]);
    sortSwap(&localData[2], &localData[6]);
    sortSwap(&localData[10], &localData[14]);
    sortSwap(&localData[3], &localData[7]);
    sortSwap(&localData[11], &localData[15]);
    sortSwap(&localData[0], &localData[8]);
    sortSwap(&localData[1], &localData[9]);
    sortSwap(&localData[2], &localData[10]);
    sortSwap(&localData[3], &localData[11]);
    sortSwap(&localData[4], &localData[12]);
    sortSwap(&localData[5], &localData[13]);
    sortSwap(&localData[6], &localData[14]);
    sortSwap(&localData[7], &localData[15]);
    sortSwap(&localData[5], &localData[10]);
    sortSwap(&localData[6], &localData[9]);
    sortSwap(&localData[3], &localData[12]);
    sortSwap(&localData[13], &localData[14]);
    sortSwap(&localData[7], &localData[11]);
    sortSwap(&localData[1], &localData[2]);
    sortSwap(&localData[4], &localData[8]);
    sortSwap(&localData[1], &localData[4]);
    sortSwap(&localData[7], &localData[13]);
    sortSwap(&localData[2], &localData[8]);
    sortSwap(&localData[11], &localData[14]);
    sortSwap(&localData[2], &localData[4]);
    sortSwap(&localData[5], &localData[6]);
    sortSwap(&localData[9], &localData[10]);
    sortSwap(&localData[11], &localData[13]);
    sortSwap(&localData[3], &localData[8]);
    sortSwap(&localData[7], &localData[12]);
    sortSwap(&localData[6], &localData[8]);
    sortSwap(&localData[10], &localData[12]);
    sortSwap(&localData[3], &localData[5]);
    sortSwap(&localData[7], &localData[9]);
    sortSwap(&localData[3], &localData[4]);
    sortSwap(&localData[5], &localData[6]);
    sortSwap(&localData[7], &localData[8]);
    sortSwap(&localData[9], &localData[10]);
    sortSwap(&localData[11], &localData[12]);
    sortSwap(&localData[6], &localData[7]);
    sortSwap(&localData[8], &localData[9]);

    for (uint32_t i = 0; i < size; i++) {
        data[i] = localData[i];
    }
}

template <typename T>
__device__ void sortingNetwork24(T *data, uint32_t size) {
    T localData[24];
    for (uint32_t i = 0; i < size; i++) {
        localData[i] = data[i];
    }
    for (uint32_t i = size; i < 24; i++) {
        localData[i] = std::numeric_limits<T>::max();
    }

    sortSwap(&localData[0], &localData[16]);
    sortSwap(&localData[1], &localData[17]);
    sortSwap(&localData[2], &localData[18]);
    sortSwap(&localData[3], &localData[19]);
    sortSwap(&localData[4], &localData[20]);
    sortSwap(&localData[5], &localData[21]);
    sortSwap(&localData[6], &localData[22]);
    sortSwap(&localData[7], &localData[23]);
    sortSwap(&localData[0], &localData[8]);
    sortSwap(&localData[1], &localData[9]);
    sortSwap(&localData[2], &localData[10]);
    sortSwap(&localData[3], &localData[11]);
    sortSwap(&localData[4], &localData[12]);
    sortSwap(&localData[5], &localData[13]);
    sortSwap(&localData[6], &localData[14]);
    sortSwap(&localData[7], &localData[15]);
    sortSwap(&localData[8], &localData[16]);
    sortSwap(&localData[9], &localData[17]);
    sortSwap(&localData[10], &localData[18]);
    sortSwap(&localData[11], &localData[19]);
    sortSwap(&localData[12], &localData[20]);
    sortSwap(&localData[13], &localData[21]);
    sortSwap(&localData[14], &localData[22]);
    sortSwap(&localData[15], &localData[23]);
    sortSwap(&localData[0], &localData[4]);
    sortSwap(&localData[1], &localData[5]);
    sortSwap(&localData[2], &localData[6]);
    sortSwap(&localData[3], &localData[7]);
    sortSwap(&localData[8], &localData[12]);
    sortSwap(&localData[9], &localData[13]);
    sortSwap(&localData[10], &localData[14]);
    sortSwap(&localData[11], &localData[15]);
    sortSwap(&localData[16], &localData[20]);
    sortSwap(&localData[17], &localData[21]);
    sortSwap(&localData[18], &localData[22]);
    sortSwap(&localData[19], &localData[23]);
    sortSwap(&localData[4], &localData[16]);
    sortSwap(&localData[5], &localData[17]);
    sortSwap(&localData[6], &localData[18]);
    sortSwap(&localData[7], &localData[19]);
    sortSwap(&localData[4], &localData[8]);
    sortSwap(&localData[5], &localData[9]);
    sortSwap(&localData[6], &localData[10]);
    sortSwap(&localData[7], &localData[11]);
    sortSwap(&localData[12], &localData[16]);
    sortSwap(&localData[13], &localData[17]);
    sortSwap(&localData[14], &localData[18]);
    sortSwap(&localData[15], &localData[19]);
    sortSwap(&localData[0], &localData[2]);
    sortSwap(&localData[1], &localData[3]);
    sortSwap(&localData[4], &localData[6]);
    sortSwap(&localData[5], &localData[7]);
    sortSwap(&localData[8], &localData[10]);
    sortSwap(&localData[9], &localData[11]);
    sortSwap(&localData[12], &localData[14]);
    sortSwap(&localData[13], &localData[15]);
    sortSwap(&localData[16], &localData[18]);
    sortSwap(&localData[17], &localData[19]);
    sortSwap(&localData[20], &localData[22]);
    sortSwap(&localData[21], &localData[23]);
    sortSwap(&localData[2], &localData[16]);
    sortSwap(&localData[3], &localData[17]);
    sortSwap(&localData[6], &localData[20]);
    sortSwap(&localData[7], &localData[21]);
    sortSwap(&localData[2], &localData[8]);
    sortSwap(&localData[3], &localData[9]);
    sortSwap(&localData[6], &localData[12]);
    sortSwap(&localData[7], &localData[13]);
    sortSwap(&localData[10], &localData[16]);
    sortSwap(&localData[11], &localData[17]);
    sortSwap(&localData[14], &localData[20]);
    sortSwap(&localData[15], &localData[21]);
    sortSwap(&localData[2], &localData[4]);
    sortSwap(&localData[3], &localData[5]);
    sortSwap(&localData[6], &localData[8]);
    sortSwap(&localData[7], &localData[9]);
    sortSwap(&localData[10], &localData[12]);
    sortSwap(&localData[11], &localData[13]);
    sortSwap(&localData[14], &localData[16]);
    sortSwap(&localData[15], &localData[17]);
    sortSwap(&localData[18], &localData[20]);
    sortSwap(&localData[19], &localData[21]);
    sortSwap(&localData[0], &localData[1]);
    sortSwap(&localData[2], &localData[3]);
    sortSwap(&localData[4], &localData[5]);
    sortSwap(&localData[6], &localData[7]);
    sortSwap(&localData[8], &localData[9]);
    sortSwap(&localData[10], &localData[11]);
    sortSwap(&localData[12], &localData[13]);
    sortSwap(&localData[14], &localData[15]);
    sortSwap(&localData[16], &localData[17]);
    sortSwap(&localData[18], &localData[19]);
    sortSwap(&localData[20], &localData[21]);
    sortSwap(&localData[22], &localData[23]);
    sortSwap(&localData[1], &localData[16]);
    sortSwap(&localData[3], &localData[18]);
    sortSwap(&localData[5], &localData[20]);
    sortSwap(&localData[7], &localData[22]);
    sortSwap(&localData[1], &localData[8]);
    sortSwap(&localData[3], &localData[10]);
    sortSwap(&localData[5], &localData[12]);
    sortSwap(&localData[7], &localData[14]);
    sortSwap(&localData[9], &localData[16]);
    sortSwap(&localData[11], &localData[18]);
    sortSwap(&localData[13], &localData[20]);
    sortSwap(&localData[15], &localData[22]);
    sortSwap(&localData[1], &localData[4]);
    sortSwap(&localData[3], &localData[6]);
    sortSwap(&localData[5], &localData[8]);
    sortSwap(&localData[7], &localData[10]);
    sortSwap(&localData[9], &localData[12]);
    sortSwap(&localData[11], &localData[14]);
    sortSwap(&localData[13], &localData[16]);
    sortSwap(&localData[15], &localData[18]);
    sortSwap(&localData[17], &localData[20]);
    sortSwap(&localData[19], &localData[22]);
    sortSwap(&localData[1], &localData[2]);
    sortSwap(&localData[3], &localData[4]);
    sortSwap(&localData[5], &localData[6]);
    sortSwap(&localData[7], &localData[8]);
    sortSwap(&localData[9], &localData[10]);
    sortSwap(&localData[11], &localData[12]);
    sortSwap(&localData[13], &localData[14]);
    sortSwap(&localData[15], &localData[16]);
    sortSwap(&localData[17], &localData[18]);
    sortSwap(&localData[19], &localData[20]);
    sortSwap(&localData[21], &localData[22]);

    for (uint32_t i = 0; i < size; i++) {
        data[i] = localData[i];
    }
}

template <typename T>
__device__ void sortingNetwork32(T *data, uint32_t size) {
    T localData[32];
    for (uint32_t i = 0; i < size; i++) {
        localData[i] = data[i];
    }
    for (uint32_t i = size; i < 32; i++) {
        localData[i] = std::numeric_limits<T>::max();
    }

    sortSwap(&localData[0], &localData[16]);
    sortSwap(&localData[1], &localData[17]);
    sortSwap(&localData[2], &localData[18]);
    sortSwap(&localData[3], &localData[19]);
    sortSwap(&localData[4], &localData[20]);
    sortSwap(&localData[5], &localData[21]);
    sortSwap(&localData[6], &localData[22]);
    sortSwap(&localData[7], &localData[23]);
    sortSwap(&localData[8], &localData[24]);
    sortSwap(&localData[9], &localData[25]);
    sortSwap(&localData[10], &localData[26]);
    sortSwap(&localData[11], &localData[27]);
    sortSwap(&localData[12], &localData[28]);
    sortSwap(&localData[13], &localData[29]);
    sortSwap(&localData[14], &localData[30]);
    sortSwap(&localData[15], &localData[31]);
    sortSwap(&localData[0], &localData[8]);
    sortSwap(&localData[1], &localData[9]);
    sortSwap(&localData[2], &localData[10]);
    sortSwap(&localData[3], &localData[11]);
    sortSwap(&localData[4], &localData[12]);
    sortSwap(&localData[5], &localData[13]);
    sortSwap(&localData[6], &localData[14]);
    sortSwap(&localData[7], &localData[15]);
    sortSwap(&localData[16], &localData[24]);
    sortSwap(&localData[17], &localData[25]);
    sortSwap(&localData[18], &localData[26]);
    sortSwap(&localData[19], &localData[27]);
    sortSwap(&localData[20], &localData[28]);
    sortSwap(&localData[21], &localData[29]);
    sortSwap(&localData[22], &localData[30]);
    sortSwap(&localData[23], &localData[31]);
    sortSwap(&localData[8], &localData[16]);
    sortSwap(&localData[9], &localData[17]);
    sortSwap(&localData[10], &localData[18]);
    sortSwap(&localData[11], &localData[19]);
    sortSwap(&localData[12], &localData[20]);
    sortSwap(&localData[13], &localData[21]);
    sortSwap(&localData[14], &localData[22]);
    sortSwap(&localData[15], &localData[23]);
    sortSwap(&localData[0], &localData[4]);
    sortSwap(&localData[1], &localData[5]);
    sortSwap(&localData[2], &localData[6]);
    sortSwap(&localData[3], &localData[7]);
    sortSwap(&localData[8], &localData[12]);
    sortSwap(&localData[9], &localData[13]);
    sortSwap(&localData[10], &localData[14]);
    sortSwap(&localData[11], &localData[15]);
    sortSwap(&localData[16], &localData[20]);
    sortSwap(&localData[17], &localData[21]);
    sortSwap(&localData[18], &localData[22]);
    sortSwap(&localData[19], &localData[23]);
    sortSwap(&localData[24], &localData[28]);
    sortSwap(&localData[25], &localData[29]);
    sortSwap(&localData[26], &localData[30]);
    sortSwap(&localData[27], &localData[31]);
    sortSwap(&localData[4], &localData[16]);
    sortSwap(&localData[5], &localData[17]);
    sortSwap(&localData[6], &localData[18]);
    sortSwap(&localData[7], &localData[19]);
    sortSwap(&localData[12], &localData[24]);
    sortSwap(&localData[13], &localData[25]);
    sortSwap(&localData[14], &localData[26]);
    sortSwap(&localData[15], &localData[27]);
    sortSwap(&localData[4], &localData[8]);
    sortSwap(&localData[5], &localData[9]);
    sortSwap(&localData[6], &localData[10]);
    sortSwap(&localData[7], &localData[11]);
    sortSwap(&localData[12], &localData[16]);
    sortSwap(&localData[13], &localData[17]);
    sortSwap(&localData[14], &localData[18]);
    sortSwap(&localData[15], &localData[19]);
    sortSwap(&localData[20], &localData[24]);
    sortSwap(&localData[21], &localData[25]);
    sortSwap(&localData[22], &localData[26]);
    sortSwap(&localData[23], &localData[27]);
    sortSwap(&localData[0], &localData[2]);
    sortSwap(&localData[1], &localData[3]);
    sortSwap(&localData[4], &localData[6]);
    sortSwap(&localData[5], &localData[7]);
    sortSwap(&localData[8], &localData[10]);
    sortSwap(&localData[9], &localData[11]);
    sortSwap(&localData[12], &localData[14]);
    sortSwap(&localData[13], &localData[15]);
    sortSwap(&localData[16], &localData[18]);
    sortSwap(&localData[17], &localData[19]);
    sortSwap(&localData[20], &localData[22]);
    sortSwap(&localData[21], &localData[23]);
    sortSwap(&localData[24], &localData[26]);
    sortSwap(&localData[25], &localData[27]);
    sortSwap(&localData[28], &localData[30]);
    sortSwap(&localData[29], &localData[31]);
    sortSwap(&localData[2], &localData[16]);
    sortSwap(&localData[3], &localData[17]);
    sortSwap(&localData[6], &localData[20]);
    sortSwap(&localData[7], &localData[21]);
    sortSwap(&localData[10], &localData[24]);
    sortSwap(&localData[11], &localData[25]);
    sortSwap(&localData[14], &localData[28]);
    sortSwap(&localData[15], &localData[29]);
    sortSwap(&localData[2], &localData[8]);
    sortSwap(&localData[3], &localData[9]);
    sortSwap(&localData[6], &localData[12]);
    sortSwap(&localData[7], &localData[13]);
    sortSwap(&localData[10], &localData[16]);
    sortSwap(&localData[11], &localData[17]);
    sortSwap(&localData[14], &localData[20]);
    sortSwap(&localData[15], &localData[21]);
    sortSwap(&localData[18], &localData[24]);
    sortSwap(&localData[19], &localData[25]);
    sortSwap(&localData[22], &localData[28]);
    sortSwap(&localData[23], &localData[29]);
    sortSwap(&localData[2], &localData[4]);
    sortSwap(&localData[3], &localData[5]);
    sortSwap(&localData[6], &localData[8]);
    sortSwap(&localData[7], &localData[9]);
    sortSwap(&localData[10], &localData[12]);
    sortSwap(&localData[11], &localData[13]);
    sortSwap(&localData[14], &localData[16]);
    sortSwap(&localData[15], &localData[17]);
    sortSwap(&localData[18], &localData[20]);
    sortSwap(&localData[19], &localData[21]);
    sortSwap(&localData[22], &localData[24]);
    sortSwap(&localData[23], &localData[25]);
    sortSwap(&localData[26], &localData[28]);
    sortSwap(&localData[27], &localData[29]);
    sortSwap(&localData[0], &localData[1]);
    sortSwap(&localData[2], &localData[3]);
    sortSwap(&localData[4], &localData[5]);
    sortSwap(&localData[6], &localData[7]);
    sortSwap(&localData[8], &localData[9]);
    sortSwap(&localData[10], &localData[11]);
    sortSwap(&localData[12], &localData[13]);
    sortSwap(&localData[14], &localData[15]);
    sortSwap(&localData[16], &localData[17]);
    sortSwap(&localData[18], &localData[19]);
    sortSwap(&localData[20], &localData[21]);
    sortSwap(&localData[22], &localData[23]);
    sortSwap(&localData[24], &localData[25]);
    sortSwap(&localData[26], &localData[27]);
    sortSwap(&localData[28], &localData[29]);
    sortSwap(&localData[30], &localData[31]);
    sortSwap(&localData[1], &localData[16]);
    sortSwap(&localData[3], &localData[18]);
    sortSwap(&localData[5], &localData[20]);
    sortSwap(&localData[7], &localData[22]);
    sortSwap(&localData[9], &localData[24]);
    sortSwap(&localData[11], &localData[26]);
    sortSwap(&localData[13], &localData[28]);
    sortSwap(&localData[15], &localData[30]);
    sortSwap(&localData[1], &localData[8]);
    sortSwap(&localData[3], &localData[10]);
    sortSwap(&localData[5], &localData[12]);
    sortSwap(&localData[7], &localData[14]);
    sortSwap(&localData[9], &localData[16]);
    sortSwap(&localData[11], &localData[18]);
    sortSwap(&localData[13], &localData[20]);
    sortSwap(&localData[15], &localData[22]);
    sortSwap(&localData[17], &localData[24]);
    sortSwap(&localData[19], &localData[26]);
    sortSwap(&localData[21], &localData[28]);
    sortSwap(&localData[23], &localData[30]);
    sortSwap(&localData[1], &localData[4]);
    sortSwap(&localData[3], &localData[6]);
    sortSwap(&localData[5], &localData[8]);
    sortSwap(&localData[7], &localData[10]);
    sortSwap(&localData[9], &localData[12]);
    sortSwap(&localData[11], &localData[14]);
    sortSwap(&localData[13], &localData[16]);
    sortSwap(&localData[15], &localData[18]);
    sortSwap(&localData[17], &localData[20]);
    sortSwap(&localData[19], &localData[22]);
    sortSwap(&localData[21], &localData[24]);
    sortSwap(&localData[23], &localData[26]);
    sortSwap(&localData[25], &localData[28]);
    sortSwap(&localData[27], &localData[30]);
    sortSwap(&localData[1], &localData[2]);
    sortSwap(&localData[3], &localData[4]);
    sortSwap(&localData[5], &localData[6]);
    sortSwap(&localData[7], &localData[8]);
    sortSwap(&localData[9], &localData[10]);
    sortSwap(&localData[11], &localData[12]);
    sortSwap(&localData[13], &localData[14]);
    sortSwap(&localData[15], &localData[16]);
    sortSwap(&localData[17], &localData[18]);
    sortSwap(&localData[19], &localData[20]);
    sortSwap(&localData[21], &localData[22]);
    sortSwap(&localData[23], &localData[24]);
    sortSwap(&localData[25], &localData[26]);
    sortSwap(&localData[27], &localData[28]);
    sortSwap(&localData[29], &localData[30]);

    for (uint32_t i = 0; i < size; i++) {
        data[i] = localData[i];
    }
}

#endif
