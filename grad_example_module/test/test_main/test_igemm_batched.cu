#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>

#include "../_simt_igemm_batched_kernels/initialize_simt_igemm_batched.h"
#include "../templates/operation.h"
#include "../error_helper.h"

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status                            \
      << "(" << cudaGetErrorString(status) << ")";                    \
      FatalError(_error.str());                                        \
    }                                                                   \
}

void initialize_igemm_batched_simt(std::vector<Operation *>&);

void find_best_algo(    
    int m,
    int n,
    int k,
    int batch_count,
    void * device_A_array,
    void * device_B_array,
    void * device_C_array) {

    std::vector<Operation *> ops;

    initialize_igemm_batched_simt(ops);

    cudaEvent_t events[2];
    checkCudaErrors(cudaEventCreate(&events[0]));
    checkCudaErrors(cudaEventCreate(&events[1]));

    float min_runtime_ms = 10000.0;
    std::string min_conf;

    for (auto& op : ops ){
        bool success = true;

        // Warm up
        for (int i = 0; i < 3; ++i) {
            cudaError_t ret = op->run(m, n, k,
                                    1.0,
                                    (int8_t **)device_A_array,
                                    k,
                                    (int8_t **)device_B_array,
                                    k,
                                    (float **)device_C_array,
                                    m,
                                    0.0,
                                    batch_count,
                                    NULL);
            
            if (ret != 0) {
                break;
                success = false;
            }
        }

        if (!success) {
            cudaDeviceSynchronize();
            continue;
        }

        checkCudaErrors(cudaEventRecord(events[0]));

        // Measure runtime_ms
        for (int i = 0; i < 20; ++i) {
            op->run(m, n, k,
                    1.0,
                    (int8_t **)device_A_array,
                    k,
                    (int8_t **)device_B_array,
                    k,
                    (float **)device_C_array,
                    m,
                    0.0,
                    batch_count,
                    NULL);
        }

        checkCudaErrors(cudaEventRecord(events[1]));
        checkCudaErrors(cudaDeviceSynchronize());

        float runtime_ms = 0.0;
        checkCudaErrors(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
        runtime_ms = runtime_ms / 20.0;

        if (runtime_ms < min_runtime_ms) {
            min_runtime_ms = runtime_ms;
            min_conf = op->name;
        }
    }

    printf("[%d, %d, %d, %d] : %f ms with (%s)\n",
        m, n, k, batch_count,
        min_runtime_ms, min_conf.c_str());
}

int main(int argc, char * argv[]) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int batch_count = atoi(argv[4]);

    std::vector<void *> host_A_array(batch_count, NULL);
    std::vector<void *> host_B_array(batch_count, NULL);
    std::vector<void *> host_C_array(batch_count, NULL);

    for (int i = 0; i < batch_count; ++i) {
        checkCudaErrors(cudaMalloc(&host_A_array[i], sizeof(int8_t)*m*k));
        checkCudaErrors(cudaMalloc(&host_B_array[i], sizeof(int8_t)*k*n));
        checkCudaErrors(cudaMalloc(&host_C_array[i], sizeof(float)*m*n));
    }
    
    void * device_A_array;
    void * device_B_array;
    void * device_C_array;
    checkCudaErrors(cudaMalloc(&device_A_array, sizeof(void *)*batch_count));
    checkCudaErrors(cudaMalloc(&device_B_array, sizeof(void *)*batch_count));
    checkCudaErrors(cudaMalloc(&device_C_array, sizeof(void *)*batch_count));
    checkCudaErrors(cudaMemcpy(device_A_array, &host_A_array[0], sizeof(void *)*batch_count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_B_array, &host_B_array[0], sizeof(void *)*batch_count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_C_array, &host_C_array[0], sizeof(void *)*batch_count, cudaMemcpyHostToDevice));

    find_best_algo(m, n, k, batch_count,
                   device_A_array,
                   device_B_array,
                   device_C_array);

    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("Time elapsed : %f s\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0);
}

