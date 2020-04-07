#ifndef SORT_ENVIRONMENT_H
#define SORT_ENVIRONMENT_H

#include <cuda_runtime_api.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "generators.h"
#include "quicksort_iterative.h"

#include "third-party/bitonic_sort.h"
#include "third-party/mergesort.h"
#include "third-party/odd_even.h"

#ifdef NO_2PHASE
#include "quicksort.h"  // Newest version
#endif

#ifndef NO_2PHASE
#include "quicksort_2phase.h"  // Old version
#endif

#ifndef NO_REC
#include "quicksort_recursive.h"
#include "third-party/cdpAdvancedQuicksort/cdpAdvancedQuicksort.h"
#include "third-party/cdpSimpleQuicksort/cdpSimpleQuicksort.h"
#endif

template <typename T>
void sort(char device, char algorithm, uint32_t n, uint32_t runs, uint32_t threshold, bool print, int verbose, double *bestCase, double *worstCase, double *average,
          void (*randomDataFunction)(uint32_t, T *), uint32_t maxBlocks = 0, uint32_t qs2Threads = 0, uint32_t elementsThread = 0, uint32_t scanThreshold = 0,
          uint32_t pivot_type = 0, bool *stoppedEarly = nullptr, double bestTime = 2.9) {
    T *data;
    cudaHostAlloc(&data, n * sizeof(T), cudaHostAllocDefault);

#ifdef NO_2PHASE
    size_t startfreeMem, endfreeMem, totalMem;
    cudaMemGetInfo(&startfreeMem, &totalMem);
#endif

    for (uint32_t r = 0; r < runs; r++) {
        // Stop early during tests
        if (stoppedEarly != nullptr && r > 0) {
            if (average[0] > bestTime * runs) {
                *stoppedEarly = true;
                gpuErrchk(cudaDeviceReset());
                cudaFreeHost(data);
                return;
            }
        }

        randomDataFunction(n, data);

        if (print) {
            std::ofstream inf("input.txt");
            if (!inf) {
                std::cerr << "Couldn't write to input.txt" << std::endl;
            }
            for (uint32_t i = 0; i < n; i++) {
                inf << data[i] << std::endl;
            }
        }

        // Sort data
        double startTime = 0, endTime = 0, callTime = 0, finishedTime = 0;

        callTime = clock();
        if (device == 'C') {
            if (algorithm == 'm') {
                startTime = clock();
                std::stable_sort(data, data + n);  // Mergesort
                endTime = clock();
            } else if (algorithm == 'q') {
                startTime = clock();  // Introsort (Quicksort/Heapsort)
                std::sort(data, data + n);
                endTime = clock();
            }
        } else {
            if (algorithm == 'a') {
#ifndef NO_REC
                run_qsort(n, data, 0, 0, verbose, startTime, endTime);
#endif
            } else if (algorithm == 'b') {
                bitonic_sort(n, data, startTime, endTime);
            } else if (algorithm == 'm') {
                mergesort(n, data, startTime, endTime);
            } else if (algorithm == 'o') {
                odd_even_caller(n, data, startTime, endTime);
            } else if (algorithm == 'q') {
#ifdef NO_2PHASE
                quicksort(n, data, startTime, endTime, endfreeMem, totalMem, threshold, maxBlocks, qs2Threads, elementsThread, scanThreshold, pivot_type);
#endif
#ifndef NO_2PHASE
                if (maxBlocks != 0 && qs2Threads != 0 && elementsThread != 0) {
                    quicksort_2phase(n, data, startTime, endTime, threshold, maxBlocks, qs2Threads, elementsThread);
                } else {
                    quicksort_2phase(n, data, startTime, endTime, threshold);
                }
#endif
            } else if (algorithm == 'i') {
                quicksort_iterative(n, data, verbose, startTime, endTime);
            } else if (algorithm == 'r') {
#ifndef NO_REC
                quicksort_recursive(n, data, verbose, startTime, endTime);
#endif
            } else if (algorithm == 's') {
#ifndef NO_REC
                qsort_caller(n, data, verbose, startTime, endTime);
#endif
            } else if (algorithm == 't') {
#ifndef NO_THRUST
                T *d_data;
                gpuErrchk(cudaMalloc(&d_data, n * sizeof(T)));
                gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(T), cudaMemcpyHostToDevice));
                gpuErrchk(cudaDeviceSynchronize());
                thrust::device_ptr<T> d_ptr(d_data);

                startTime = clock();
                thrust::sort(d_ptr, d_ptr + n);

                gpuErrchk(cudaDeviceSynchronize());
                endTime = clock();
                gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
                cudaFree(d_data);
#endif
            }
        }
        finishedTime = clock();

        // Check for errors
        if (verbose) {
            T previous = data[0];
            for (uint32_t i = 0; i < n; i++) {
                if (i != 0 && previous > data[i]) {
                    std::cout << "Wrong sorted on value " << data[i] << " at index " << i << "; previous = " << previous << std::endl;
                }
                previous = data[i];
            }
        }

        if (print) {
            std::ofstream outf("output.txt");
            if (!outf) {
                std::cerr << "Couldn't write to output.txt" << std::endl;
            }
            for (uint32_t i = 0; i < n; i++) {
                outf << data[i] << std::endl;
            }
        }

        double sortTime = (endTime - startTime) / CLOCKS_PER_SEC;
        double totalTime = (finishedTime - callTime) / CLOCKS_PER_SEC;

        if (r == 0) {
            average[0] = sortTime;
            average[1] = totalTime;
            bestCase[0] = sortTime;
            worstCase[0] = sortTime;
            bestCase[1] = totalTime;
            worstCase[1] = totalTime;
        } else {
            bestCase[0] = (bestCase[0] < sortTime) ? bestCase[0] : sortTime;
            worstCase[0] = (worstCase[0] > sortTime) ? worstCase[0] : sortTime;
            bestCase[1] = (bestCase[1] < totalTime) ? bestCase[1] : totalTime;
            worstCase[1] = (worstCase[1] > totalTime) ? worstCase[1] : totalTime;
            average[0] += sortTime;
            average[1] += totalTime;
        }
    }

#ifdef NO_2PHASE
    if (algorithm == 'q' && verbose) {
        std::cout << "Memory:\n"
                  << "  Total:        " << totalMem / 1048576 << "MiB\n"
                  << "  Total used:   " << (totalMem - endfreeMem) / 1048576 << "MiB\n"
                  << "  Used:         " << ((totalMem - endfreeMem) - (totalMem - startfreeMem)) / 1048576 << "MiB\n"
                  << "  Difference:   " << (totalMem - startfreeMem) / 1048576 << "MiB" << std::endl;
    }
#endif

    average[0] /= runs;
    average[1] /= runs;

    if (verbose) {
        std::cout << "\nFinished sorting:\n"
                     "  Sorting time:\n"
                  << "      Best case:   " << std::to_string(bestCase[0]) << "s\n"
                  << "      Worst case:  " << std::to_string(worstCase[0]) << "s\n"
                  << "      Average:     " << std::to_string(average[0]) << "s\n"
                  << "  Total time:\n"
                  << "      Best case:   " << std::to_string(bestCase[1]) << "s\n"
                  << "      Worst case:  " << std::to_string(worstCase[1]) << "s\n"
                  << "      Average:     " << std::to_string(average[1]) << "s\n"
                  << std::endl;
    }

    cudaFreeHost(data);
}
#endif