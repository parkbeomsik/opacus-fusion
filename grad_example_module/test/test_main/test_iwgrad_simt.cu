#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>

#include "../_simt_iwgrad_kernels/initialize_simt_iwgrad_splitK.h"
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

void initialize_conv2d_wgrad_simt(std::vector<Operation *>&);

void find_best_algo(    
    int N,
    int H,
    int W,
    int C,
    int K,
    int R,
    int S,
    int P,
    int Q,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    void * ograd,
    void * actv,
    void * wgrad,
    void * ws) {

    std::vector<Operation *> ops;
    std::vector<int> split_k_slices_cand = {2, 4, 8, 16, 32, 64};

    initialize_conv2d_wgrad_simt(ops);

    cudaEvent_t events[2];
    checkCudaErrors(cudaEventCreate(&events[0]));
    checkCudaErrors(cudaEventCreate(&events[1]));

    float min_runtime_ms = 10000.0;
    std::string min_conf;
    int min_split_k;

    for (auto& op : ops ){
        for (auto split_k_slices : split_k_slices_cand) {
            bool success = true;

            // Warm up
            for (int i = 0; i < 3; ++i) {
                cudaError_t ret = op->run((int8_t *)ograd,
                                        (int8_t *)actv,
                                        (float *)wgrad,
                                        ws,
                                        N,
                                        H,
                                        W,
                                        C,
                                        K,
                                        R,
                                        S,
                                        P,
                                        Q,
                                        pad_h,
                                        pad_w,
                                        stride_h,
                                        stride_w,
                                        dilation_h,
                                        dilation_w,
                                        split_k_slices,
                                        1.0,
                                        0.0,
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
                op->run((int8_t *)ograd,
                        (int8_t *)actv,
                        (float *)wgrad,
                        ws,
                        N,
                        H,
                        W,
                        C,
                        K,
                        R,
                        S,
                        P,
                        Q,
                        pad_h,
                        pad_w,
                        stride_h,
                        stride_w,
                        dilation_h,
                        dilation_w,
                        split_k_slices,
                        1.0,
                        0.0,
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
                min_split_k = split_k_slices;
            }
        }
    }

    printf("[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d] : %f ms with (%s, splitK=%d)\n",
        N, H, W, C, K, R, S, P, Q,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        min_runtime_ms, min_conf.c_str(), min_split_k);
}

int main(int argc, char * argv[]) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int N = atoi(argv[1]);
    int H = atoi(argv[2]);
    int W = atoi(argv[3]);
    int C = atoi(argv[4]);
    int K = atoi(argv[5]);
    int R = atoi(argv[6]);
    int S = atoi(argv[7]);
    int P = atoi(argv[8]);
    int Q = atoi(argv[9]);
    int pad_h = atoi(argv[10]);
    int pad_w = atoi(argv[11]);
    int stride_h = atoi(argv[12]);
    int stride_w = atoi(argv[13]);
    int dilation_h = atoi(argv[14]);
    int dilation_w = atoi(argv[15]);

    size_t ws_size = 1 * (size_t(1) << 30);

    void * ograd;
    void * actv;
    void * wgrad;
    void * ws;
    checkCudaErrors(cudaMalloc(&ograd, sizeof(int8_t)*N*K*P*Q));
    checkCudaErrors(cudaMalloc(&actv, sizeof(int8_t)*N*C*H*W));
    checkCudaErrors(cudaMalloc(&wgrad, sizeof(float)*K*C*R*S));
    checkCudaErrors(cudaMalloc(&ws, ws_size));

    find_best_algo(    
                    N,
                    H,
                    W,
                    C,
                    K,
                    R,
                    S,
                    P,
                    Q,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    ograd,
                    actv,
                    wgrad,
                    ws);

    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("Time elapsed : %f s\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0);
}

