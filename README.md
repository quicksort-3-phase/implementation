#  Quicksort 3-phase

Code of Quicksort 3-phase and other algorithms tested in the paper
"Quicksort 3-phase - Parallel sorting on graphics cards programmed in CUDA."

This package consists of the following files:
* generators.h: generates the datasets for the test cases
* helper.h: contains swap functions, sorting networks and error handlers
* main.cu: runs the different test cases and sorting algorithms
* makefile
* quicksort.h: interface for Quicksort 3-phase
* quicksort_2phase.h: interface for Quicksort 2-phase
* quicksort_iterative.h: interface for Quicksort iterative
* quicksort_recursive.h: interface for Quicksort recursive
* sort_environment.h: interface for the algorithms and includes time recording
* structs.h: contains the data types Record & Vector
* third-party
    * cdpAdvancedQuicksort
        * cdpAdvancedQuicksort.h: interface for cdpAdvancedQuicksort by [NVIDIA](https://github.com/NVIDIA/cuda-samples)
        * cdpBitonicSort.cu
        * cdpQuicksort.h
    * cdpSimpleQuicksort
        * cdpSimpleQuicksort.h: interface for cdpSimpleQuicksort by [NVIDIA](https://github.com/NVIDIA/cuda-samples)
    * bitonic_sort.h: interface for Bitonic sort by [Mahir Jain](https://github.com/mahirjain25/Parallel-Sorting-Algorithms)
    * helper_cuda.h
    * helper_string.h
    * mergesort.h: interface for Mergesort by [Mahir Jain](https://github.com/mahirjain25/Parallel-Sorting-Algorithms)
    * odd_even.h: interface for Odd–even sort by [Mahir Jain](https://github.com/mahirjain25/Parallel-Sorting-Algorithms)


### How to build
Compile with the following flags:
```
nvcc main.cu -O2 -arch=sm_35 -rdc=true --expt-relaxed-constexpr
```
or you can simply use:
```
make release
```

### How to run
Usage with the arguments:
```
./a.out [device] [data type] <algorithm <amount>> [special case] [runs <amount>] [threshold <amount>] [verbose] [print]
```
Where the following options are provided:

* device:
    * C - CPU
    * G - GPU (default)

* data types:
    * D - Double
    * F - Float
    * I - Integer
    * Y - Record
    * V - Vector

* algorithms:
    * b - Bitonic sort
    * m - Mergesort
    * o - Odd Even sort
    * q - Quicksort
    * i - Quicksort iterative
    * r - Quicksort recursive
    * s - Simple Quicksort
    * t - Thrust sort
    * A - All (comparison test)
    * Q - Quicksort (combination test)
    * P - Quicksort (pivot test)

* special case:
    * X - Inverse sorted
    * H - Half sorted
    * S - Sorted
    * Z - All equal

* runs:
    * R - Default one run

* threshold:
    * T - Default equals 32 (or one for Quicksort 2-phase)

* verbose:
    * v - Default equals one

* print:
    * p - Print out input and output file

## Authors
* [Joel Bienias](https://github.com/bieniajl) | bieniajl@fius.informatik.uni-stuttgart.de
* [Alexander Fischer](https://github.com/infality/) | st149038@stud.uni-stuttgart.de
* [Rene Richard Tischler](https://github.com/st149535/) | st149535@stud.uni-stuttgart.de
* [Faris Uhlig](https://github.com/farisu) | faris.uhlig@outlook.de
