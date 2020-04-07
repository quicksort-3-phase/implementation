#ifndef QUICKSORT_RECURSIVE_H
#define QUICKSORT_RECURSIVE_H

#include <math.h>
#include <thrust/sort.h>

#include <ctime>
#include <iostream>
#include <random>
#include <string>

#define REC_DEPTH 24

// recursive partition (hoare scheme)
template <typename T>
__device__ uint64_t partition_recursive_hoare(uint64_t l, uint64_t r, T *a) {
    // printf("%i-%i, ", l, r);
    uint64_t middle = (l + r) / 2;

    // Choose pivot (median of three)
    if (a[l] > a[middle]) {
        swap(&a[l], &a[middle]);
    }
    if (a[l] > a[r]) {
        swap(&a[l], &a[r]);
    }
    if (a[middle] > a[r]) {
        swap(&a[middle], &a[r]);
    }

    T pivot = a[middle];
    uint64_t i = l;
    uint64_t j = r;
    while (true) {
        do {
            i++;
        } while (a[i] < pivot);
        do {
            j--;
        } while (a[j] > pivot);
        if (i >= j) {
            return j;
        }
        swap(&a[i], &a[j]);
    }
}

// recursive partition (lomuto scheme)
template <typename T>
__device__ uint64_t partition_recursive_lomuto(uint64_t l, uint64_t r, T *a) {
    // printf("%i-%i, ", l, r);
    uint64_t middle = (l + r) / 2;

    // Choose pivot (median of three and store median in r)
    if (a[l] > a[middle]) {
        swap(&a[l], &a[middle]);
    }
    if (a[l] > a[r]) {
        swap(&a[l], &a[r]);
    }
    if (a[r] > a[middle]) {
        swap(&a[r], &a[middle]);
    }

    T pivot = a[r];
    // Actual partitioning
    uint64_t i = l;
    for (uint64_t j = l; j < r; j++) {
        if (a[j] <= pivot) {
            swap(&a[i], &a[j]);
            i++;
        }
    }

    swap(&a[i], &a[r]);
    return i;
}

// quicksort kernel that sorts recursively (with dynamic parallelism) and falls back to thrust sort
template <typename T>
__global__ void quicksortKernel_recursive(int depth, uint64_t l, uint64_t r, T *a) {
    if (depth < REC_DEPTH) {
        uint64_t cut = partition_recursive_hoare(l, r, a);
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

        if (l < cut) {
            quicksortKernel_recursive<<<1, 1, 1, s1>>>(depth + 1, l, cut, a);
            cdpErrchk(cudaGetLastError());
        }
        if (cut + 1 < r) {
            quicksortKernel_recursive<<<1, 1, 1, s2>>>(depth + 1, cut + 1, r, a);
            cdpErrchk(cudaGetLastError());
        }
        cudaDeviceSynchronize();
    } else {
        thrust::sort(thrust::seq, &a[l], &a[r+1]);
    }
}

// function to copy memory and start recursive quicksort
template <typename T>
int quicksort_recursive(uint64_t n, T *h_data, int verbose, double &startTime, double &endTime) {
    if (n > 1) {
        T *d_data;

        if (verbose) {
            std::cout << "Copying data to device..." << std::endl;
        }

        gpuErrchk(cudaMalloc(&d_data, n * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();

        if (verbose) {
            std::cout << "Sorting data..." << std::endl;
        }
        startTime = clock();
        quicksortKernel_recursive<<<1, 1>>>(0, 0, n - 1, d_data);

        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        endTime = clock();

        if (verbose) {
            std::cout << "Copying data to host..." << std::endl;
        }
        gpuErrchk(cudaMemcpy(h_data, d_data, n * sizeof(T), cudaMemcpyDeviceToHost));

        cudaFree(d_data);
    }
    return 0;
}

#endif
