#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>

//$f(x) = (\frac{x}{x-2.3})^3$ to an array of size 75341
float* naive_map(float *arr, int n, float *result, int result_size) {
    if (result_size < n) {
        printf("Result size is less than n\n");
        assert(0);
    }
    for (int i = 0; i < n; i++) {
        float x = arr[i];
        float temp = x / (x - 2.3);
        result[i] = temp * temp * temp;
    }
}

int main() {
    int n_times = 100;

    float total_time = 0;

    // Seed the random number generator
    srand(time(NULL));

    for (int i = 0; i < n_times; i++) {
        float arr[75341];
        float result[75341];
        for (int j = 0; j < 75341; j++) {
            // Generate random float between 0 and 100
            arr[j] = (float)rand() / RAND_MAX * 100.0f;
        }

        clock_t start = clock();
        naive_map(arr, 75341, result, 75341);
        clock_t end = clock();

        double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        total_time += cpu_time_used / n_times;

        // Comment out or remove the following loop to avoid printing all results
        // for (int i = 0; i < 75341; i++) {
        //     printf("%f\n", result[i]);
        // }
    }
    printf("Average execution time: %f seconds\n", total_time);
    return 0;
}