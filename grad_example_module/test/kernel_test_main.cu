
#include "templates/conv2d_wgrad_tensorop.h"
#include "templates/operation.h"
#include "test_wgrad_main.h"
#include "error_helper.h"

int main(int argc, char * argv[]) {
    std::vector<Operation *> ops;

    initialize_conv2d_wgrad_tensorop(ops);

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
    int split_k_slices = atoi(argv[16]);

    size_t ws_size = 10 * (size_t(1) << 30);

    void * ograd;
    void * actv;
    void * wgrad;
    void * ws;
    checkCudaErrors(cudaMalloc(&ograd, sizeof(int8_t)*N*K*P*Q));
    checkCudaErrors(cudaMalloc(&actv, sizeof(int8_t)*N*C*H*W));
    checkCudaErrors(cudaMalloc(&wgrad, sizeof(float)*K*C*R*S));
    checkCudaErrors(cudaMalloc(&ws, ws_size));

    cudaEvent_t events[2];
    checkCudaErrors(cudaEventCreate(&events[0]));
    checkCudaErrors(cudaEventCreate(&events[1]));

    for (auto& op : ops ){
        // Warm up
        for (int i = 0; i < 3; ++i) {
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

        printf("%f\n", runtime_ms);
    }
}

