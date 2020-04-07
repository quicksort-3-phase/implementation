#ifndef STRUCTS_H
#define STRUCTS_H

struct Record {
    int key;
    int value[2];

    __host__ __device__ Record(){};
    __host__ __device__ Record(int v) : key(v) { value[v % 2] = v; };
};

__host__ __device__ bool operator<(const Record& lhs, const Record& rhs) { return lhs.key < rhs.key; }

__host__ __device__ bool operator>(const Record& lhs, const Record& rhs) { return lhs.key > rhs.key; }

__host__ __device__ bool operator<=(const Record& lhs, const Record& rhs) { return lhs.key <= rhs.key; }

__host__ __device__ bool operator>=(const Record& lhs, const Record& rhs) { return lhs.key >= rhs.key; }

__host__ __device__ bool operator==(const Record& lhs, const Record& rhs) { return lhs.key == rhs.key; }

__host__ __device__ bool operator!=(const Record& lhs, const Record& rhs) { return lhs.key != rhs.key; }

std::ostream& operator<<(std::ostream& out, const Record& lhs) {
    out << lhs.key;
    return out;
}

template <>
__host__ __device__ Record std::numeric_limits<Record>::max() {
    return Record(2147483647);
};

struct Vector {
    double components[2];

    __host__ __device__ Vector(){};
    __host__ __device__ Vector(int v) {
        for (int i = 0; i < 2; i++) components[i] = v;
    };
};

__host__ __device__ bool operator<(const Vector& lhs, const Vector& rhs) {
    double abs_lhs = 0, abs_rhs = 0;
    for (int i = 0; i < 2; i++) {
        abs_lhs += lhs.components[i] * lhs.components[i];
        abs_rhs += rhs.components[i] * rhs.components[i];
    }
    return abs_lhs < abs_rhs;
}

__host__ __device__ bool operator>(const Vector& lhs, const Vector& rhs) { return rhs < lhs; }

__host__ __device__ bool operator<=(const Vector& lhs, const Vector& rhs) { return !(lhs > rhs); }

__host__ __device__ bool operator>=(const Vector& lhs, const Vector& rhs) { return !(lhs < rhs); }

__host__ __device__ bool operator==(const Vector& lhs, const Vector& rhs) {
    double abs_lhs = 0, abs_rhs = 0;
    for (int i = 0; i < 2; i++) {
        abs_lhs += lhs.components[i] * lhs.components[i];
        abs_rhs += rhs.components[i] * rhs.components[i];
    }
    return abs_lhs == abs_rhs;
}

__host__ __device__ bool operator!=(const Vector& lhs, const Vector& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& out, const Vector& lhs) {
    double abs_lhs = 0;
    for (int i = 0; i < 2; i++) {
        abs_lhs += lhs.components[i] * lhs.components[i];
    }
    out << abs_lhs;
    return out;
}

template <>
__host__ __device__ Vector std::numeric_limits<Vector>::max() {
    return Vector(2147483647);
};
#endif
