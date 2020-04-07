#ifndef SIMPLE_QUICKSORT_H
#define SIMPLE_QUICKSORT_H

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <cstdio>
#include <iostream>
#include "../helper_cuda.h"
#include "../helper_string.h"

#define MAX_DEPTH 16
#define INSERTION_SORT 32

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ void selection_sort(T *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        T min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j) {
            T val_j = data[j];

            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void cdp_simple_quicksort(T *data, int left, int right, int depth) {
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    T *lptr = data + left;
    T *rptr = data + right;
    T pivot = data[(left + right) / 2];

    // Do the partitioning.
    while (lptr <= rptr) {
        // Find the next left- and right-hand values to swap
        T lval = *lptr;
        T rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot) {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot) {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr) {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr - data)) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr - data) < right) {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
template <typename T>
void run_qsort(T *data, int verbose, unsigned int nitems) {
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    int left = 0;
    int right = nitems - 1;
    if (verbose) {
        std::cout << "Launching kernel on the GPU" << std::endl;
    }
    cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
    checkCudaErrors(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
// Entry point for main function.
////////////////////////////////////////////////////////////////////////////////
template <typename T>
int qsort_caller(uint64_t num_items, T *h_data, int verbose, double &startTime, double &endTime) {
    // Create input data
    T *d_data;

    // Allocate GPU memory.
    if (verbose) {
        std::cout << "Copying data to device..." << std::endl;
    }
    checkCudaErrors(cudaMalloc((void **)&d_data, num_items * sizeof(T)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(T), cudaMemcpyHostToDevice));

    // Execute
    if (verbose) {
        std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    }
    startTime = clock();
    run_qsort(d_data, verbose, num_items);

    gpuErrchk(cudaDeviceSynchronize());
    endTime = clock();

    if (verbose) {
        std::cout << "Copying data to host..." << std::endl;
    }
    gpuErrchk(cudaMemcpy(h_data, d_data, num_items * sizeof(T), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_data));
    return 0;
}
#endif
