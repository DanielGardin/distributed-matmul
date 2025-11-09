#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <time.h>

#include "matrix.h"
#include "serial_matmul.h"

void print_usage() {
    printf("Usage: ./test_performance -s <size> -n <num_tests>\n");
    printf("  -s <size>       Size of the NxN matrices to multiply (default: 512)\n");
}

int main(int argc, char *argv[]) {
    int opt;
    int size = 512;

    extern char *optarg;

    while( (opt = getopt(argc, argv, "s:n:")) > 0 ) {
        switch(opt) {
            case 's':
                size = atoi(optarg);
                if (size <= 0) {
                    fprintf(stderr, "Invalid matrix size. Size must be positive.\n");
                    return 1;
                }
                break;

            case 'h':
                print_usage();
                return 0;

            default:
                print_usage();
                return 1;
        }
    }
    
    printf("Starting performance tests with matrix size %dx%d...\n", size, size);

    long long int mem = (long long int) 3 * size * size * sizeof(float);

    if (mem < 1024) {
        printf("Requesting a total of %lld bytes of memory.", mem);

    } else if ((mem >> 10) < 1024) {
        printf("Requesting a total of %.2f kb of memory.", mem/1024.);

    } else if ((mem >> 20) < 1024) {
        printf("Requesting a total of %.2f Mb of memory.", mem/(1024.*1024));

    } else if ((mem >> 30) < 1024) {
        printf("Requesting a total of %.2f Gb of memory.", mem/(1024.*1024*1024));

    }
    puts("\n");

    
    Matrix A = create_matrix(size, size);
    Matrix B = create_matrix(size, size);
    Matrix C = create_matrix(size, size);
    
    
    
    srand(0);
    for (int i = 0; i < A.rows * A.cols; i++) {
        A.data[i] = 2 * ((float) rand() / RAND_MAX) - 1;
        B.data[i] = 2 * ((float) rand() / RAND_MAX) - 1;
    }

    // Measure performance of serial matrix multiplication
    clock_t start = clock();
    serial_matmul(&A, &B, &C);
    clock_t end = clock();
    
    if (C.data) {
        double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Serial multiplication took %.6f seconds.\n", time_taken);
    } else {
        fprintf(stderr, "Matrix multiplication failed.\n");
    }

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);
    
    return 0;
}