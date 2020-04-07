#include <getopt.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#define RUNS 10

// Comment to enable assert.h
#define NDEBUG

// Comment to enable recursion
// #define NO_REC

// Comment to enable thrust
// #define NO_THRUST

// Comment to enable old version
#define NO_2PHASE

#ifndef NO_THRUST
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

#include "generators.h"
#include "helper.h"
#include "sort_environment.h"
#include "structs.h"

std::ofstream createCSV(std::string path, std::string *filename, int i, int year, int month, int day) {
    std::string file;
    if (month - 10 < 0 && day - 10 < 0) {
        file = filename[i] + std::to_string(year) + "0" + std::to_string(month) + "0" + std::to_string(day) + ".csv";
    } else if (month - 10 < 0) {
        file = filename[i] + std::to_string(year) + "0" + std::to_string(month) + std::to_string(day) + ".csv";
    } else if (day - 10 < 0) {
        file = filename[i] + std::to_string(year) + "0" + std::to_string(month) + "0" + std::to_string(day) + ".csv";
    } else {
        file = filename[i] + std::to_string(year) + std::to_string(month) + std::to_string(day) + ".csv";
    }
    std::ofstream ofs(path + file);
    if (!ofs) {
        std::cerr << "Couldn't write to " << file << std::endl;
    }
    return ofs;
}

int main(int argc, char *argv[]) {
    bool print = false;
    int c, cdpCapable, cuda_device, day, month, year,
        max_exponent = 27, max_elements = pow(2, max_exponent),  // 134217728 - Largest possible value for GTX 1060 6GB, old value was 16777216.
        start = 4, verbose = 1;
    struct tm *timeinfo;
    uint32_t n, runs = 1, sc = 0, threads[] = {128, 256, 512, 768, 1024}, thresholds[] = {1, 16, 24, 32};
    time_t rawtime;
    char algorithm, data_type,
        device = 'G', devAlgo[][2] = {{'C', 'm'}, {'C', 'q'}, {'G', 'a'}, {'G', 'b'}, {'G', 'm'}, {'G', 'o'}, {'G', 'q'}, {'G', 'i'}, {'G', 'r'}, {'G', 't'}, {'G', 's'}};

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    year = timeinfo->tm_year + 1900;
    month = timeinfo->tm_mon + 1;
    day = timeinfo->tm_mday;

    std::ofstream csv;
    std::string algo[] = {"std::stable_sort", "std::sort",           "cdpAdvancedQuicksort", "Bitonic sort", "Mergesort",         "Odd Even sort",
                          "Quicksort",        "Quicksort iterative", "Quicksort recursive",  "Thrust sort",  "cdpSimpleQuicksort"},
                dev[] = {"CPU", "GPU"}, filename[] = {"test_",      "test_half_sorted_", "test_inverse_sorted_",   "test_sorted_",
                                                      "test_zero_", "combination_test_", "combination_test_qs2p_", "pivot_test_"},
                input[][2] = {{"doubles", "Double"}, {"floats", "Float"}, {"integers", "Integer"}, {"records", "Record"}, {"vectors", "Vector"}}, path = "./results/",
                pivot_types[] =
                    {"median-of-three",        "median-of-five",  "median-of-three-medians-of-three", "median-of-three-medians-of-five", "median-of-five-medians-of-five",
                     "median-of-twenty-three", "median-of-sqrt-n"},
                special_case[] = {" with ", " with half sorted ", " with inverse sorted ", " with sorted ", " with equal "};

    void (*doubleGenerator[])(uint32_t, double *) = {randomDoubles, halfSortedDoubles, inverseSortedDoubles, sortedDoubles, zeroDoubles};
    void (*floatGenerator[])(uint32_t, float *) = {randomFloats, halfSortedFloats, inverseSortedFloats, sortedFloats, zeroFloats};
    void (*integerGenerator[])(uint32_t, int *) = {randomIntegers, halfSortedIntegers, inverseSortedIntegers, sortedIntegers, zeroIntegers};
    void (*recordGenerator[])(uint32_t, Record *) = {randomRecords, halfSortedRecords, inverseSortedRecords, sortedRecords, zeroRecords};
    void (*vectorGenerator[])(uint32_t, Vector *) = {randomVectors, halfSortedVectors, inverseSortedVectors, sortedVectors, zeroVectors};

    /* Sorting time: 0, Total time: 1 */
    double bestCase[2] = {}, worstCase[2] = {}, average[2] = {};

#ifdef NO_2PHASE
    uint32_t threshold = 32;
#endif

#ifndef NO_2PHASE
    uint32_t threshold = 1;
#endif

    // Argument validation
    if (argc < 2) {
        std::cout << "Usage: ./a.out [device] [data type] <algorithm <amount>> [special case]\n"
                  << "               [runs <amount>] [threshold <amount>] [verbose] [print]\n\n"
                  << "device:       C - CPU\n"
                  << "              G - GPU (default)\n\n"
                  << "data types:   D - Double\n"
                  << "              F - Float\n"
                  << "              I - Integer\n"
                  << "              Y - Record\n\n"
                  << "              V - Vector\n\n"
                  << "algorithms:   a - Advanced Quicksort\n"
                  << "              b - Bitonic sort\n"
                  << "              m - Mergesort\n"
                  << "              o - Odd Even sort\n"
                  << "              q - Quicksort\n"
                  << "              i - Quicksort iterative\n"
                  << "              r - Quicksort recursive\n"
                  << "              s - Simple Quicksort\n"
                  << "              t - Thrust sort\n"
                  << "              A - All (comparison test)\n"
                  << "              Q - Quicksort (combination test)\n"
                  << "              P - Quicksort (pivot test)\n\n"
                  << "special case: X - Inverse sorted\n"
                  << "              H - Half sorted\n"
                  << "              S - Sorted\n"
                  << "              Z - All equal\n\n"
                  << "runs:         R - Default one run\n"
                  << "threshold:    T - Default equals 32 (or one for Quicksort 2-phase)\n"
                  << "verbose:      v - Default equals one\n\n"
                  << "print:        p - Print out input and output file" << std::endl;
        return 1;
    }

    option long_options[] = {{"advanced-quicksort", 1, 0, 'a'},
                             {"quicksort", 1, 0, 'q'},
                             {"quicksort-iterative", 1, 0, 'i'},
                             {"quicksort-recursive", 1, 0, 'r'},
                             {"bitonic-sort", 1, 0, 'm'},
                             {"mergesort", 1, 0, 'b'},
                             {"odd-even-sort", 1, 0, 'o'},
                             {"simple-quicksort", 1, 0, 's'},
                             {"thrust", 1, 0, 't'},
                             {"all", 1, 0, 'A'},
                             {0, 0, 0, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    while (1) {
        c = getopt_long(argc, argv, "a:q:i:r:m:b:o:s:t:R:T:v:pACDFGHIQPSVXYZ", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0) break;
                printf("option %s", long_options[option_index].name);
                if (optarg) printf(" with arg %s", optarg);
                printf("\n");
                break;

            case 'a':
                n = atoi(optarg);
                algorithm = 'a';
                break;

            case 'q':
                n = atoi(optarg);
                algorithm = 'q';
                break;

            case 'i':
                n = atoi(optarg);
                algorithm = 'i';
                break;

            case 'r':
                n = atoi(optarg);
                algorithm = 'r';
                break;

            case 'b':
                n = atoi(optarg);
                algorithm = 'b';
                break;

            case 'm':
                n = atoi(optarg);
                algorithm = 'm';
                break;

            case 'o':
                n = atoi(optarg);
                algorithm = 'o';
                break;
            case 'p':
                print = true;
                break;

            case 's':
                n = atoi(optarg);
                algorithm = 's';
                break;

            case 't':
                n = atoi(optarg);
                algorithm = 't';
                break;

            case 'v':
                verbose = atoi(optarg);
                break;

            case 'A':
                data_type = 'A';
                break;

            case 'C':
                device = 'C';
                break;

            case 'D':
                data_type = 'D';
                break;

            case 'F':
                data_type = 'F';
                break;

            case 'G':
                device = 'G';
                break;

            case 'H':
                sc = 1;
                break;

            case 'I':
                data_type = 'I';
                break;

            case 'Q':
                data_type = 'Q';
                break;

            case 'P':
                data_type = 'P';
                break;

            case 'R':
                runs = atoi(optarg);
                break;

            case 'S':
                sc = 3;
                break;

            case 'T':
                threshold = atoi(optarg);
                break;

            case 'V':
                data_type = 'V';
                break;

            case 'X':
                sc = 2;
                break;

            case 'Y':
                data_type = 'Y';
                break;

            case 'Z':
                sc = 4;
                break;

            case '?':
                /* getopt_long already printed an error message. */
                break;

            default:
                abort();
        }
    }
    // Get device properties
    cuda_device = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp properties;
    checkCudaErrors(cudaGetDeviceProperties(&properties, cuda_device));
    cdpCapable = (properties.major == 3 && properties.minor >= 5) || properties.major >= 4;
    std::cout << "GPU device " << properties.name << " has compute capabilities (SM " << properties.major << "." << properties.minor << ")" << std::endl;

    if (!cdpCapable) {
        std::cout << "Requires SM 3.5 or higher to use CUDA Dynamic Parallelism." << std::endl;
        return 1;
    } else if (algorithm == 0 && !(data_type == 'A' || data_type == 'P' || data_type == 'Q')) {
        std::cout << "Please enter an algorithm." << std::endl;
        return 1;
    } else if (data_type == 'A') {
        verbose = 0;

        csv = createCSV(path, filename, sc, year, month, day);
        csv << "Algorithm, Elements, Data_Type, sT_BestCase, sT_WorstCase, sT_Average, tT_BestCase, tT_WorstCase, tT_Average" << std::endl;

        for (int i = 0; i < 11; i++) {
            device = devAlgo[i][0];
            algorithm = devAlgo[i][1];

            for (int j = 0; j < 5; j++) {
                if (i == 5) {
                    max_elements = 524288;
                } else if (i == 10) {
                    max_elements = 4194304;
                } else if (j > 2) {
                    max_elements = 67108864;
                } else {
                    max_elements = pow(2, max_exponent);
                }

                if (i < 2) {
                    std::cout << "Running " << algo[i] << special_case[sc] << input[j][0] << " on " << dev[0] << std::endl;
                } else {
                    std::cout << "Running " << algo[i] << special_case[sc] << input[j][0] << " on " << dev[1] << std::endl;
                }

                if (j == 0) {
                    for (int k = 0; pow(2, k) <= max_elements; k++) {
                        runs = (int)pow(2, (RUNS - 0.245 * k));
                        sort<double>(device, algorithm, pow(2, k), runs, threshold, print, verbose, bestCase, worstCase, average, doubleGenerator[sc]);
                        csv << algo[i] << ", " << (int)pow(2, k) << ", " << input[j][1] << ", " << std::fixed << bestCase[0] << ", " << worstCase[0] << ", " << average[0]
                            << ", " << bestCase[1] << ", " << worstCase[1] << ", " << average[1] << std::endl;
                    }
                } else if (j == 1) {
                    for (int k = 0; pow(2, k) <= max_elements; k++) {
                        runs = (int)pow(2, (RUNS - 0.245 * k));
                        sort<float>(device, algorithm, pow(2, k), runs, threshold, print, verbose, bestCase, worstCase, average, floatGenerator[sc]);
                        csv << algo[i] << ", " << (int)pow(2, k) << ", " << input[j][1] << ", " << std::fixed << bestCase[0] << ", " << worstCase[0] << ", " << average[0]
                            << ", " << bestCase[1] << ", " << worstCase[1] << ", " << average[1] << std::endl;
                    }
                } else if (j == 2) {
                    for (int k = 0; pow(2, k) <= max_elements; k++) {
                        runs = (int)pow(2, (RUNS - 0.245 * k));
                        sort<int>(device, algorithm, pow(2, k), runs, threshold, print, verbose, bestCase, worstCase, average, integerGenerator[sc]);
                        csv << algo[i] << ", " << (int)pow(2, k) << ", " << input[j][1] << ", " << std::fixed << bestCase[0] << ", " << worstCase[0] << ", " << average[0]
                            << ", " << bestCase[1] << ", " << worstCase[1] << ", " << average[1] << std::endl;
                    }
                } else if (j == 3) {
                    for (int k = 0; pow(2, k) <= max_elements; k++) {
                        runs = (int)pow(2, (RUNS - 0.245 * k));
                        sort<Record>(device, algorithm, pow(2, k), runs, threshold, print, verbose, bestCase, worstCase, average, recordGenerator[sc]);
                        csv << algo[i] << ", " << (int)pow(2, k) << ", " << input[j][1] << ", " << std::fixed << bestCase[0] << ", " << worstCase[0] << ", " << average[0]
                            << ", " << bestCase[1] << ", " << worstCase[1] << ", " << average[1] << std::endl;
                    }
                } else {
                    for (int k = 0; pow(2, k) <= max_elements; k++) {
                        runs = (int)pow(2, (RUNS - 0.245 * k));
                        sort<Vector>(device, algorithm, pow(2, k), runs, threshold, print, verbose, bestCase, worstCase, average, vectorGenerator[sc]);
                        csv << algo[i] << ", " << (int)pow(2, k) << ", " << input[j][1] << ", " << std::fixed << bestCase[0] << ", " << worstCase[0] << ", " << average[0]
                            << ", " << bestCase[1] << ", " << worstCase[1] << ", " << average[1] << std::endl;
                    }
                }
            }
        }

    } else if (data_type == 'D') {
        sort<double>(device, algorithm, n, runs, threshold, print, verbose, bestCase, worstCase, average, doubleGenerator[sc]);
    } else if (data_type == 'F') {
        sort<float>(device, algorithm, n, runs, threshold, print, verbose, bestCase, worstCase, average, floatGenerator[sc]);
    } else if (data_type == 'I') {
        sort<int>(device, algorithm, n, runs, threshold, print, verbose, bestCase, worstCase, average, integerGenerator[sc]);
    } else if (data_type == 'Y') {
        sort<Record>(device, algorithm, n, runs, threshold, print, verbose, bestCase, worstCase, average, recordGenerator[sc]);
    } else if (data_type == 'V') {
        sort<Vector>(device, algorithm, n, runs, threshold, print, verbose, bestCase, worstCase, average, vectorGenerator[sc]);
    } else if (data_type == 'Q') {
        algorithm = devAlgo[6][1];
        verbose = 0;
        runs = 10;

#ifdef NO_2PHASE
        csv = createCSV(path, filename, 5, year, month, day);
        bool stoppedEarly = false;
        uint32_t size = max_exponent - (start - 1);
        double bestAverageTimes[size];
        for (uint32_t i = 0; i < size; i++) {
            bestAverageTimes[i] = 2.9;
        }
#endif

#ifndef NO_2PHASE
        csv = createCSV(path, filename, 6, year, month, day);
#endif

#ifdef NO_2PHASE
        csv << "Max_Blocks, Threads, Elements_Thread, Scan_Threshold, Threshold, Elements, Data_Type, sT_BestCase, sT_WorstCase, sT_Average, tT_BestCase, "
               "tT_WorstCase, tT_Average"
            << std::endl;
#endif

#ifndef NO_2PHASE
        csv << "Max_Blocks, Threads, Elements_Thread, Threshold, Elements, Data_Type, sT_BestCase, sT_WorstCase, sT_Average, tT_BestCase, tT_WorstCase, tT_Average"
            << std::endl;
#endif

        // Blocks - Min: 2^5/32 | Max: 2^15/32768
        for (int b = 5; b <= 15; b++) {
            // Threads - 2^7/128, 2^8/256, 2^9/512, 2^9.585/768 2^10/1024
            for (int t = 0; t < 5; t++) {
                // Elements per Thread - Min: 64 | Max: 2048
                for (int i = 6; i <= 11; i++) {
#ifdef NO_2PHASE
                    // ScanThreshold - Min: 128 | Max: 4096
                    for (int s = 7; s <= 12; s++) {
                        // Threshold - 1, 16, 24, 32
                        for (int th = 0; th < 4; th++) {
#endif
                            // Input - Double | Float | Integer
                            // for (int j = 0; j <= 2; j++) {
#ifdef NO_2PHASE
                            std::cout << "Running " << algo[6] << special_case[0] << pow(2, b) << " Blocks, " << threads[t] << " Threads, " << pow(2, i)
                                      << " Elements per Thread, Scan Threshold of " << pow(2, s) << " and Threshold of " << thresholds[th] << " on random " << input[0][0]
                                      << std::endl;
#endif

#ifndef NO_2PHASE
                            std::cout << "Running " << algo[6] << special_case[0] << pow(2, b) << " Blocks, " << threads[t] << " Threads, " << pow(2, i)
                                      << " Elements per Thread and Threshold of " << threshold << " on random " << input[0][0] << std::endl;
#endif

                            // Elements - Min: 1024 | Max: 16777216
                            for (int k = start; pow(2, k) <= max_elements; k++) {
                                // runs = (int)pow(2, (RUNS - 0.245 * k));
#ifdef NO_2PHASE
                                stoppedEarly = false;
                                sort<double>(device, algorithm, pow(2, k), runs, thresholds[th], print, verbose, bestCase, worstCase, average, doubleGenerator[sc],
                                             pow(2, b), threads[t], pow(2, i), pow(2, s), 0, &stoppedEarly, bestAverageTimes[k - start]);
                                if (!stoppedEarly) {
                                    csv << (int)pow(2, b) << ", " << threads[t] << ", " << (int)pow(2, i) << ", " << (int)pow(2, s) << ", " << thresholds[th] << ", "
                                        << (int)pow(2, k) << ", " << input[0][1] << ", " << std::fixed << bestCase[0] << ", " << worstCase[0] << ", " << average[0]
                                        << ", " << bestCase[1] << ", " << worstCase[1] << ", " << average[1] << std::endl;
                                    if (average[0] < bestAverageTimes[k - start]) {
                                        bestAverageTimes[k - start] = average[0];
                                    }
                                }
#endif

#ifndef NO_2PHASE
                                sort<double>(device, algorithm, pow(2, k), runs, threshold, print, verbose, bestCase, worstCase, average, doubleGenerator[sc], pow(2, b),
                                             threads[t], pow(2, i));
                                csv << (int)pow(2, b) << ", " << threads[t] << ", " << (int)pow(2, i) << ", " << threshold << ", " << (int)pow(2, k) << ", "
                                    << input[0][1] << ", " << std::fixed << bestCase[0] << ", " << worstCase[0] << ", " << average[0] << ", " << bestCase[1] << ", "
                                    << worstCase[1] << ", " << average[1] << std::endl;
#endif
                            }
                            //}
#ifdef NO_2PHASE
                        }
                    }
#endif
                }
            }
        }
    } else if (data_type == 'P') {
        algorithm = devAlgo[6][1];
        verbose = 0;

        csv = createCSV(path, filename, 7, year, month, day);
        csv << "Algorithm, Pivot_Type, Elements, Data_Type, sT_BestCase, sT_WorstCase, sT_Average, tT_BestCase, tT_WorstCase, tT_Average" << std::endl;

        for (int pt = 1; pt < 6; pt++) {
            std::cout << "Running " << algo[6] << special_case[0] << "pivot type: " << pivot_types[pt - 1] << " on random " << input[0][0] << std::endl;
            for (int k = 0; pow(2, k) <= max_elements; k++) {
                runs = (int)pow(2, (RUNS - 0.245 * k));
                sort<double>(device, algorithm, pow(2, k), runs, 1, print, verbose, bestCase, worstCase, average, doubleGenerator[sc], 0, 0, 0, 0, pt);
                csv << algo[6] << ", " << pivot_types[pt - 1] << ", " << (int)pow(2, k) << ", " << input[0][1] << ", " << std::fixed << bestCase[0] << ", "
                    << worstCase[0] << ", " << average[0] << ", " << bestCase[1] << ", " << worstCase[1] << ", " << average[1] << std::endl;
            }
        }
    } else {
        std::cout << "Please enter a data type option to sort!" << std::endl;
    }
    return 0;
}
