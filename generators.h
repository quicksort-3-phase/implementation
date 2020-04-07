#ifndef GENERATORS_H
#define GENERATORS_H

#include <cfloat>
#include <climits>
#include <limits>
#include <random>

#include "helper.h"
#include "structs.h"

static std::mt19937 rng(std::random_device{}());

void randomDoubles(uint32_t n, double* data) {
    std::uniform_real_distribution<double> dis(DBL_MIN, DBL_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }
}

void randomFloats(uint32_t n, float* data) {
    std::uniform_real_distribution<float> dis(FLT_MIN, FLT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }
}

void randomIntegers(uint32_t n, int* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }
}

void randomRecords(uint32_t n, Record* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = Record(dis(rng));
    }
}

void randomVectors(uint32_t n, Vector* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = Vector(dis(rng));
    }
}

void zeroDoubles(uint32_t n, double* data) {
    std::uniform_real_distribution<double> dis(DBL_MIN, DBL_MAX);
    double rn = dis(rng);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = rn;
    }
}

void zeroFloats(uint32_t n, float* data) {
    std::uniform_real_distribution<float> dis(FLT_MIN, FLT_MAX);
    float rn = dis(rng);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = rn;
    }
}

void zeroIntegers(uint32_t n, int* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    int rn = dis(rng);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = rn;
    }
}

void zeroRecords(uint32_t n, Record* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    Record rn = Record(dis(rng));
    for (uint32_t i = 0; i < n; i++) {
        data[i] = rn;
    }
}

void zeroVectors(uint32_t n, Vector* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    Vector rn = Vector(dis(rng));
    for (uint32_t i = 0; i < n; i++) {
        data[i] = rn;
    }
}

void halfSortedDoubles(uint32_t n, double* data) {
    std::uniform_real_distribution<double> dis(DBL_MIN, DBL_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    double* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<double> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + (n / 2));
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + (n / 2));
#endif
}

void halfSortedFloats(uint32_t n, float* data) {
    std::uniform_real_distribution<float> dis(FLT_MIN, FLT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    float* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<float> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + (n / 2));
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + (n / 2));
#endif
}

void halfSortedIntegers(uint32_t n, int* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    int* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<int> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + (n / 2));
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + (n / 2));
#endif
}

void halfSortedRecords(uint32_t n, Record* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = Record(dis(rng));
    }

#ifndef NO_THRUST
    Record* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(Record)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(Record), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<Record> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + (n / 2));
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(Record), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + (n / 2));
#endif
}

void halfSortedVectors(uint32_t n, Vector* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = Vector(dis(rng));
    }

#ifndef NO_THRUST
    Vector* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(Vector)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(Vector), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<Vector> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + (n / 2));
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(Vector), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + (n / 2));
#endif
}

void inverseSortedDoubles(uint32_t n, double* data) {
    std::uniform_real_distribution<double> dis(DBL_MIN, DBL_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    double* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<double> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n, thrust::greater<double>());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n, std::greater<double>());
#endif
}

void inverseSortedFloats(uint32_t n, float* data) {
    std::uniform_real_distribution<float> dis(FLT_MIN, FLT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    float* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<float> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n, thrust::greater<float>());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n, std::greater<float>());
#endif
}

void inverseSortedIntegers(uint32_t n, int* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    int* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<int> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n, thrust::greater<int>());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n, std::greater<int>());
#endif
}

void inverseSortedRecords(uint32_t n, Record* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = Record(dis(rng));
    }

#ifndef NO_THRUST
    Record* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(Record)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(Record), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<Record> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n, thrust::greater<Record>());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(Record), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n, std::greater<Record>());
#endif
}

void inverseSortedVectors(uint32_t n, Vector* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = Vector(dis(rng));
    }

#ifndef NO_THRUST
    Vector* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(Vector)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(Vector), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<Vector> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n, thrust::greater<Vector>());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(Vector), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n, std::greater<Vector>());
#endif
}

void sortedDoubles(uint32_t n, double* data) {
    std::uniform_real_distribution<double> dis(DBL_MIN, DBL_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    double* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<double> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n);
#endif
}

void sortedFloats(uint32_t n, float* data) {
    std::uniform_real_distribution<float> dis(FLT_MIN, FLT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    float* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<float> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n);
#endif
}

void sortedIntegers(uint32_t n, int* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = dis(rng);
    }

#ifndef NO_THRUST
    int* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<int> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n);
#endif
}

void sortedRecords(uint32_t n, Record* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = Record(dis(rng));
    }

#ifndef NO_THRUST
    Record* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(Record)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(Record), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<Record> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(Record), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n);
#endif
}

void sortedVectors(uint32_t n, Vector* data) {
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX);
    for (uint32_t i = 0; i < n; i++) {
        data[i] = Vector(dis(rng));
    }

#ifndef NO_THRUST
    Vector* d_data;
    gpuErrchk(cudaMalloc(&d_data, n * sizeof(Vector)));
    gpuErrchk(cudaMemcpy(d_data, data, n * sizeof(Vector), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    thrust::device_ptr<Vector> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + n);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(data, d_data, n * sizeof(Vector), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
#endif

#ifdef NO_THRUST
    std::sort(data, data + n);
#endif
}
#endif
