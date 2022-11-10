#include <vector>
// #include "base_operation.h"
// #include "cutlass_error.h"
#include "cutlass_wgrad_grouped.h"
#include "initialize_swgrad_grouped.h"

#if defined(_USE_TENSOR_CORE)
#include "initialize_iwgrad_tensorop_grouped.h"
#else
#include "initialize_iwgrad_grouped.h"
#endif

#include "wgrad_grouped_operation.h"

#include "cuda_runtime.h"
// #include "cuda_error_helper.h"

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/layout/tensor.h"


namespace cutlass_wgrad_grouped {

void * _device_problems = NULL;

std::vector<Operation *> operations;

std::vector<void *> device_workspaces;
// std::vector<void *> host_workspaces;

std::vector<OperationWithWorkspace> operations_with_workspaces;

void ** device_ptr_A;
void ** device_ptr_B;
void ** device_ptr_C;
void ** device_ptr_D;

int problem_count;

#if defined(_USE_TENSOR_CORE)
void initialize_int_tensorop() {
    initialize_iwgrad_tensorop_grouped(operations);
}
#else
void initialize_int() {

    initialize_iwgrad_grouped(operations);
}
#endif

void initialize_float() {

    initialize_swgrad_grouped(operations);
}

template<typename dType>
void initialize_problems(std::vector<Conv2dConfig> const & host_configs) {

    using namespace cutlass::conv;
    
    problem_count = host_configs.size();
    // printf("problem count = %d\n", problem_count);

    // Set problem sizes in host memory first
    std::vector<Conv2dProblemSize> host_problems;
    for (int i = 0; i < problem_count; ++i) {
        Conv2dConfig host_config = host_configs.at(i);

        // Set single problem in host
        Conv2dProblemSize problem(host_config.N, 
                                  host_config.H, host_config.W, host_config.C,
                                  host_config.K, host_config.R, host_config.S,
                                  host_config.P, host_config.Q,
                                  host_config.pad_h, host_config.pad_w,
                                  host_config.stride_h, host_config.stride_w,
                                  host_config.dilation_h, host_config.dilation_w,
                                  Mode::kCrossCorrelation, host_config.split_k_slices);

        host_problems.push_back(problem);
    }

    assert(host_problems.size() == problem_count);

    // Set problems in device memory
    if (_device_problems != NULL) {
        checkCudaErrors(cudaFree(_device_problems));
    }
    checkCudaErrors(cudaMalloc(&_device_problems, 
                                (size_t)sizeof(Conv2dProblemSize)*problem_count));
    checkCudaErrors(cudaMemcpy((void *)_device_problems, (void *)host_problems.data(), 
                               (size_t)sizeof(Conv2dProblemSize)*problem_count,
                               cudaMemcpyHostToDevice));

    // Set problems of operations
    // Allocate host tensor ref
    using namespace cutlass::layout;
    using TensorRefIn = cutlass::TensorRef<dType, cutlass::layout::TensorNHWC>;
    using TensorRefOut = cutlass::TensorRef<float, cutlass::layout::TensorNHWC>;
    std::vector<TensorRefIn> host_ref_A(problem_count);
    std::vector<TensorRefIn> host_ref_B(problem_count);
    std::vector<TensorRefOut> host_ref_C(problem_count);
    std::vector<TensorRefOut> host_ref_D(problem_count);
    // Initialize host tensor refs
    for (int i = 0; i < problem_count; ++i) {
        Conv2dConfig conf = host_configs.at(i);

        host_ref_A.at(i).reset(NULL, TensorNHWC(TensorNHWC::packed(cutlass::Tensor4DCoord(conf.N, conf.P, conf.Q, conf.K))));
        host_ref_B.at(i).reset(NULL, TensorNHWC(TensorNHWC::packed(cutlass::Tensor4DCoord(conf.N, conf.H, conf.W, conf.C))));
        host_ref_C.at(i).reset(NULL, TensorNHWC(TensorNHWC::packed(cutlass::Tensor4DCoord(conf.K, conf.R, conf.S, conf.C))));
        host_ref_D.at(i).reset(NULL, TensorNHWC(TensorNHWC::packed(cutlass::Tensor4DCoord(conf.K, conf.R, conf.S, conf.C))));
    }

    // Allocate device tensor ref
    TensorRefIn *device_ref_A;
    TensorRefIn *device_ref_B;
    TensorRefOut *device_ref_C;
    TensorRefOut *device_ref_D;
    checkCudaErrors(cudaMalloc(&device_ref_A, sizeof(TensorRefIn)*problem_count));
    checkCudaErrors(cudaMalloc(&device_ref_B, sizeof(TensorRefIn)*problem_count));
    checkCudaErrors(cudaMalloc(&device_ref_C, sizeof(TensorRefOut)*problem_count));
    checkCudaErrors(cudaMalloc(&device_ref_D, sizeof(TensorRefOut)*problem_count));

    // Copy host refs to device
    checkCudaErrors(cudaMemcpy(device_ref_A, &host_ref_A[0], sizeof(TensorRefIn)*problem_count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ref_B, &host_ref_B[0], sizeof(TensorRefIn)*problem_count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ref_C, &host_ref_C[0], sizeof(TensorRefOut)*problem_count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ref_D, &host_ref_D[0], sizeof(TensorRefOut)*problem_count, cudaMemcpyHostToDevice));

    for (auto device_workspace : device_workspaces){
        checkCudaErrors(cudaFree(device_workspace));
    }
    device_workspaces.clear();
    device_workspaces.push_back(device_ref_A);
    device_workspaces.push_back(device_ref_B);
    device_workspaces.push_back(device_ref_C);
    device_workspaces.push_back(device_ref_D);

    wGradGroupedConfig wgrad_config = {(Conv2dProblemSize *)_device_problems,
                                       problem_count,
                                       (void *)device_ref_A,
                                       (void *)device_ref_B,
                                       (void *)device_ref_C,
                                       (void *)device_ref_D,
                                       &host_problems[0]};
    for (auto operation : operations) {
        void * host_workspace = malloc(operation->get_host_workspace_size());
        operations_with_workspaces.push_back(OperationWithWorkspace({operation, host_workspace}));
    }

    size_t max_workspace_size = 0;
    for (auto& operation_with_workspace : operations_with_workspaces) {
        size_t workspace_size = operation_with_workspace.operation->get_workspace_size(&wgrad_config, operation_with_workspace.host_workspace);
        if (workspace_size > max_workspace_size) {
            max_workspace_size = workspace_size;
        }
    }
    
    void * device_semaphore;
    printf("Allocate %lu B device workspace\n", max_workspace_size);
    checkCudaErrors(cudaMalloc(&device_semaphore, max_workspace_size));
    device_workspaces.push_back(device_semaphore);

    // for (auto operation_with_workspace : operations_with_workspaces) {
    //     checkCUTLASS(operation_with_workspace.operation->initialize(&wgrad_config, device_semaphore, operation_with_workspace.host_workspace));
    // }
    operations_with_workspaces.erase(std::remove_if(
        operations_with_workspaces.begin(), operations_with_workspaces.end(), 
        [&](OperationWithWorkspace op) { 
            return op.operation->initialize(&wgrad_config, device_semaphore, op.host_workspace) != cutlass_wgrad_grouped::Status::kSuccess; }), 
            operations_with_workspaces.end());

    printf("%lu operations initialized\n", operations_with_workspaces.size());

    // assert(operations_with_workspaces.size() == operations.size());
}

template void initialize_problems<int8_t>(std::vector<Conv2dConfig> const &);
template void initialize_problems<float>(std::vector<Conv2dConfig> const &);

void finalize() {
    for (auto device_workspace : device_workspaces){
        checkCudaErrors(cudaFree(device_workspace));
    }
    for (auto operation_with_workspace : operations_with_workspaces) {
        free(operation_with_workspace.host_workspace);
        free(operation_with_workspace.operation);
    }
    
    device_workspaces.clear();
    operations_with_workspaces.clear();
    operations.clear();

    checkCudaErrors(cudaFree(_device_problems));
    _device_problems = NULL;
}

OperationWithWorkspace get_best_operation(void ** ptr_A,
                              void ** ptr_B,
                              void ** ptr_C,
                              void ** ptr_D) {

    // assert(operations_with_workspaces.size() == operations.size());

    std::vector<float> runtime_ms_list(operations_with_workspaces.size(), 100000.0);
    // runtime_ms_list.resize(operations.size());
    std::cout << "Get best operation..." << std::endl;

    cudaEvent_t events[2];

    for (auto & event : events) {
        checkCudaErrors(cudaEventCreate(&event));
    }

    for (int i = 0; i < operations_with_workspaces.size(); ++i) {

        auto operation = operations_with_workspaces.at(i).operation;
        auto host_workspace = operations_with_workspaces.at(i).host_workspace;

        checkCUTLASS(operation->update_ptrs(ptr_A, ptr_B, ptr_C, ptr_D, problem_count, host_workspace));

        // Warm up

        // Record an event at the start of a series of GEMM operations
        checkCudaErrors(cudaEventRecord(events[0]));

        Status result;
        for (int iter = 0; iter < 3; ++iter) {
            result = operation->run(host_workspace);
        }

        cudaError_t ret = cudaEventRecord(events[1]);

        if (ret == 0) {
            ret = cudaEventSynchronize(events[1]);
        }

        float runtime_ms;
        // Check operation success
        if (ret == 0) {
            checkCudaErrors(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));

            // checkCudaErrors(cudaDeviceSynchronize());

            if (result == Status::kSuccess && runtime_ms < 30.0) {
                // Measure runtime

                // Record an event at the start of a series of GEMM operations
                checkCudaErrors(cudaEventRecord(events[0]));

                for (int iter = 0; iter < 20; ++iter) {
                    result = operation->run(host_workspace);
                }

                checkCudaErrors(cudaEventRecord(events[1]));
                checkCudaErrors(cudaEventSynchronize(events[1]));

                checkCudaErrors(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
            }
            else {
                runtime_ms = 2000.0;
            }
        }
        else {
            runtime_ms = 2000.0;
        }

        if (result == Status::kSuccess) {
            // If CUTLASS is so slow for this problem, it stops early.
            // printf("runtime_ms = %f\t%s\n", runtime_ms, operation->name.c_str());
            if (runtime_ms > 3000.0) {
                break;
            }
            runtime_ms_list.at(i) = runtime_ms;
        } else {
            // printf("runtime_ms = %f\t%s\n", runtime_ms, operation->name.c_str());
            // Means it failed
            runtime_ms_list.at(i) = 100000.0;
        }
        // 
    }

    for (auto & event : events) {
        checkCudaErrors(cudaEventDestroy(event));
    }

    assert(runtime_ms_list.size() == operations_with_workspaces.size());

    float min_runtime_ms = 10000.0;
    OperationWithWorkspace best_operation {NULL, NULL};
    for (int i = 0; i < runtime_ms_list.size(); ++i) {
        if (runtime_ms_list.at(i) < min_runtime_ms) {
            min_runtime_ms = runtime_ms_list.at(i);
            best_operation = operations_with_workspaces.at(i);
        }
    }

    std::cout << best_operation.operation->name << "( " << min_runtime_ms / 20 << " ms )" << std::endl;

    return best_operation;
}

Status run(OperationWithWorkspace operation_with_workspace) {
    return operation_with_workspace.operation->run(operation_with_workspace.host_workspace);
}

Status update_ptrs(OperationWithWorkspace operation_with_workspace,
                   void ** ptr_A,
                   void ** ptr_B,
                   void ** ptr_C,
                   void ** ptr_D,
                   int problem_count) {
    return operation_with_workspace.operation->update_ptrs(ptr_A, ptr_B, ptr_C, ptr_D, problem_count, operation_with_workspace.host_workspace);
}


} // namespace cutlass_wgrad_grouped