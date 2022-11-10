#include <torch/extension.h>
#include "ATen/cudnn/Handles.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <vector>
#include <chrono>
#include <algorithm>
#include <unistd.h>

#include "structure.h"
#include "error_helper.h"

#include "cutlass_simt_int8_wgrad.h"

#include "cutlass_wgrad_grouped.h"

#include "compute_scaling_factor_cuda.h"

#include "grad_example_module_conv.h"
#include "utils.h"
#include "add_noise.h"

#define THRESHOLD_INCREASE_COUNT_NON_CUDNN 4
#define THRESHOLD_INCREASE_COUNT_NON_REWEIGHT 10

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global variables

bool _first_run = true;

std::vector<Conv2dDescriptor> descriptors;

int64_t partial_per_example_gradient_size = 0;
size_t non_reweight_per_example_gradient_size = 0;
size_t best_end_non_reweight_layer = 3;
size_t best_end_cudnn_layer = 3;

size_t max_bwd_filter_batch_algo_best_workspace_size = 0;
torch::Tensor tensor_for_bwd_filter_batch_workspace;
void * bwd_filter_batch_workspace = NULL;


size_t total_ws_size = 0;
size_t total_per_batch_cudnn_workspace = 0;

#define _N_CUDA_STREAMS 10
cudaStream_t cuda_streams[_N_CUDA_STREAMS];
cudnnHandle_t cudnn_handles[_N_CUDA_STREAMS];
cublasHandle_t cublas_handles[1];

// CUTLASS device workspaces
torch::Tensor tensor_for_device_ptr_A;
torch::Tensor tensor_for_device_ptr_B;
torch::Tensor tensor_for_device_ptr_C;
torch::Tensor tensor_for_device_ptr_D;
void ** device_ptr_A = NULL;
void ** device_ptr_B = NULL;
void ** device_ptr_C = NULL;
void ** device_ptr_D = NULL;

// CUTLASS host workspaces to initialze device pointers
// torch::Tensor tensor_for_host_ptr_A;
// torch::Tensor tensor_for_host_ptr_B;
// torch::Tensor tensor_for_host_ptr_C;
// torch::Tensor tensor_for_host_ptr_D;
// void ** host_ptr_A = NULL;
// void ** host_ptr_B = NULL;
// void ** host_ptr_C = NULL;
// void ** host_ptr_D = NULL;
std::vector<void *> host_ptr_A;
std::vector<void *> host_ptr_B;
std::vector<void *> host_ptr_C;
std::vector<void *> host_ptr_D;

cutlass_wgrad_grouped::OperationWithWorkspace best_operation_with_workspace;

int num_rows_to_compute = 4;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_descriptors_conv(std::vector<Conv2dConfig> &configs, bool quant=false, int end_non_reweight_layer=3) {
  // Create custom cuda streams
  for (int i=0; i < _N_CUDA_STREAMS; ++i) {
    checkCudaErrors(cudaStreamCreate(&cuda_streams[i]));
    checkCUDNN(cudnnCreate(&cudnn_handles[i]));
    checkCUDNN(cudnnSetStream(cudnn_handles[i], cuda_streams[i]));
  }
  checkCUBLAS(cublasCreate(&cublas_handles[0]));
  checkCUBLAS(cublasSetStream(cublas_handles[0], cuda_streams[0]));
  cublasSetPointerMode(cublas_handles[0], CUBLAS_POINTER_MODE_DEVICE);

  descriptors.clear();
  descriptors.resize(configs.size());

  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;

  if ((configs.at(0).H <= 32) && configs.size() <= 40) {
    num_rows_to_compute = 16;
  }
  else if((configs.at(0).H <= 32) && configs.size() <= 70) {
    num_rows_to_compute = 4;
  } 
  else { 
    num_rows_to_compute = 1;
  }
  
  for (size_t i=0; i < configs.size(); ++i) {
    auto config = &configs.at(i);
    // Generate cuDNN descriptors
    // Setup Conv2D backward filter
    descriptors.at(i).grad_weight_per_example_size_in_bytes += (size_t)sizeof(float)\
                                        *config->K\
                                        *config->C\
                                        *config->R\
                                        *config->S;

    descriptors.at(i).config = {config->N, config->H, config->W,
                            config->C, config->K, config->R, config->S,
                            config->P, config->Q, 
                            config->pad_h, config->pad_w,
                            config->stride_h, config->stride_w,
                            config->dilation_h, config->dilation_w};

    // Initialize convDesc for cudnn
    checkCUDNN(cudnnCreateConvolutionDescriptor(&descriptors.at(i).conv_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(descriptors.at(i).conv_desc,
                                              config->pad_h,
                                              config->pad_w,
                                              config->stride_h,
                                              config->stride_w,
                                              config->dilation_h,
                                              config->dilation_w,
                                              CUDNN_CROSS_CORRELATION,
                                              dtype));

    // checkCUDNN(cudnnSetConvolutionReorderType(descriptors.at(i).conv_desc,
    //                                           CUDNN_NO_REORDER));

    // Initialize tensorDesc, filter_desc and bias_desc
    checkCUDNN(cudnnCreateFilterDescriptor(&descriptors.at(i).filter_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(descriptors.at(i).filter_desc,
                                        dtype,
                                        CUDNN_TENSOR_NHWC,
                                        config->K,
                                        config->C,
                                        config->R,
                                        config->S));

    checkCUDNN(cudnnCreateTensorDescriptor(&descriptors.at(i).input_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(descriptors.at(i).input_desc,
                                        CUDNN_TENSOR_NHWC,
                                        dtype,
                                        1,
                                        config->C,
                                        config->H,
                                        config->W));

    checkCUDNN(cudnnCreateTensorDescriptor(&descriptors.at(i).output_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(descriptors.at(i).output_desc,
                                        CUDNN_TENSOR_NHWC,
                                        dtype,
                                        1,
                                        config->K,
                                        config->P,
                                        config->Q));

    checkCUDNN(cudnnCreateFilterDescriptor(&descriptors.at(i).filter_batch_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(descriptors.at(i).filter_batch_desc,
                                        dtype,
                                        CUDNN_TENSOR_NCHW,
                                        config->K,
                                        config->C,
                                        config->R,
                                        config->S));

    checkCUDNN(cudnnCreateTensorDescriptor(&descriptors.at(i).input_batch_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(descriptors.at(i).input_batch_desc,
                                        CUDNN_TENSOR_NCHW,
                                        dtype,
                                        config->N,
                                        config->C,
                                        config->H,
                                        config->W));

    checkCUDNN(cudnnCreateTensorDescriptor(&descriptors.at(i).output_batch_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(descriptors.at(i).output_batch_desc,
                                        CUDNN_TENSOR_NCHW,
                                        dtype,
                                        config->N,
                                        config->K,
                                        config->P,
                                        config->Q));

    // Find a fastest algorithm for per-example
    torch::Tensor temp_x = torch::empty({config->C*config->H*config->W}, torch::TensorOptions().device(torch::kCUDA, 0));
    torch::Tensor temp_y = torch::empty({config->K*config->P*config->Q}, torch::TensorOptions().device(torch::kCUDA, 0));
    torch::Tensor temp_w = torch::empty({config->K*config->C*config->R*config->S}, torch::TensorOptions().device(torch::kCUDA, 0));
    torch::Tensor temp_ws = torch::empty({config->C*config->H*config->W}, torch::TensorOptions().device(torch::kCUDA, 0));
    int temp_ws_size = (size_t)sizeof(float)*config->C*config->H*config->W;
    int returned_algo_count = 0;

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo_perf[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(at::native::getCudnnHandle(),
                                                            descriptors.at(i).input_desc,
                                                            descriptors.at(i).output_desc,
                                                            descriptors.at(i).conv_desc,
                                                            descriptors.at(i).filter_desc,
                                                            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                                                            &returned_algo_count,
                                                            bwd_filter_algo_perf));
    int best_idx = 0;
    // for (best_idx = 0; best_idx < CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT; ++best_idx) {
    //   if (bwd_filter_algo_perf[best_idx].determinism == CUDNN_DETERMINISTIC) {
    //     break;
    //   }
    // }

    descriptors.at(i).bwd_filter_algo_best = bwd_filter_algo_perf[best_idx].algo;
    descriptors.at(i).bwd_filter_algo_best_workspace_size = bwd_filter_algo_perf[best_idx].memory;
    
    total_ws_size += bwd_filter_algo_perf[best_idx].memory;
    descriptors.at(i).grad_weight_per_example_size_in_bytes = (size_t)sizeof(float)\
                                                                *config->K\
                                                                *config->C\
                                                                *config->R\
                                                                *config->S;
    descriptors.at(i).grad_weight_per_example_size = (size_t)config->K*config->C*config->R*config->S;
    descriptors.at(i).workspace_size_in_bytes = bwd_filter_algo_perf[best_idx].memory;
    total_ws_size += bwd_filter_algo_perf[best_idx].memory;
    partial_per_example_gradient_size += (int64_t)config->K*config->C*config->R*config->S;

    descriptors.at(i).filter_shape = {config->K, config->C, config->R, config->S};

    if (quant) {
      descriptors.at(i).workspace_size_in_bytes = \
      cutlass_simt_iwgrad_get_workspace(
                                        config->N,
                                        config->H,
                                        config->W,
                                        config->C,
                                        config->K,
                                        config->R,
                                        config->S,
                                        config->P,
                                        config->Q,
                                        config->pad_h,
                                        config->pad_w,
                                        config->stride_h,
                                        config->stride_w,
                                        config->dilation_h,
                                        config->dilation_w);
    }

    //// CUTLASS
    descriptors.at(i).cutlass_config = {1, config->H, config->W,
                                        config->C, config->K, config->R, config->S,
                                        config->P, config->Q, 
                                        config->pad_h, config->pad_w,
                                        config->stride_h, config->stride_w,
                                        config->dilation_h, config->dilation_w,
                                        config->split_k_slices};

  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_cudnn_per_batch_algorithm(std::vector<torch::Tensor>& actvs,
                                   std::vector<torch::Tensor>& ograds) {
  for (size_t i = 0; i < descriptors.size(); ++i) {
    // Find a fastest algorithm for per-batch
    auto config = descriptors.at(i).config;

    int returned_algo_count = 0;
    int temp_ws_size = (size_t)sizeof(float)*config.K*config.C*config.R*config.S;
    torch::Tensor temp_ws = torch::empty({config.K*config.C*config.R*config.S}, torch::TensorOptions().device(torch::kCUDA, 0));
    torch::Tensor wgrad = torch::empty({config.K*config.C*config.R*config.S}, torch::TensorOptions().device(torch::kCUDA, 0));

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo_perf[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    
    checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(at::native::getCudnnHandle(),
                                                            descriptors.at(i).input_batch_desc,
                                                            descriptors.at(i).output_batch_desc,
                                                            descriptors.at(i).conv_desc,
                                                            descriptors.at(i).filter_batch_desc,
                                                            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                                                            &returned_algo_count,
                                                            bwd_filter_algo_perf));
    int best_idx = 0;
    // for (best_idx = 0; best_idx < CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT; ++best_idx) {
    //   if (bwd_filter_algo_perf[best_idx].determinism == CUDNN_DETERMINISTIC && bwd_filter_algo_perf[best_idx].memory < (size_t)1024*1024*1024*1024) {
    //     break;
    //   }
    // }

    descriptors.at(i).bwd_filter_batch_algo_best = bwd_filter_algo_perf[best_idx].algo;
    descriptors.at(i).bwd_filter_batch_algo_best_workspace_size = bwd_filter_algo_perf[best_idx].memory;
    total_per_batch_cudnn_workspace += bwd_filter_algo_perf[best_idx].memory;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_cudnn_workspace(size_t end_cudnn_layer) {
  for (size_t i=0; i < descriptors.size(); ++i) {
    descriptors.at(i).tensor_for_workspace_ptr = torch::empty({0});
    descriptors.at(i).workspace_ptr = NULL;
    if (i < end_cudnn_layer) {
      if(descriptors.at(i).workspace_size_in_bytes > 0) {
        descriptors.at(i).tensor_for_workspace_ptr = torch::empty({(int64_t)descriptors.at(i).workspace_size_in_bytes}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
        descriptors.at(i).workspace_ptr = descriptors.at(i).tensor_for_workspace_ptr.data_ptr();
      }
      else {
        descriptors.at(i).workspace_ptr = NULL;
      }
    }
  } 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_per_batch_cudnn_workspace(size_t end_non_reweight_layer) {
  for (size_t i=0; i < descriptors.size(); ++i) {
    descriptors.at(i).tensor_for_batch_workspace_ptr = torch::empty({0});
    descriptors.at(i).batch_workspace_ptr = NULL;
    if (i >= end_non_reweight_layer) {
      if(descriptors.at(i).bwd_filter_batch_algo_best_workspace_size > 0) {
        descriptors.at(i).tensor_for_batch_workspace_ptr = torch::empty({(int64_t)descriptors.at(i).bwd_filter_batch_algo_best_workspace_size}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
        descriptors.at(i).batch_workspace_ptr = descriptors.at(i).tensor_for_batch_workspace_ptr.data_ptr();
      }
      else {
        descriptors.at(i).batch_workspace_ptr = NULL;
      }
    }
  } 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_cutlass_workspace(size_t end_cudnn_layer, int batch_size, bool quant=false) {
  // Set device workspace which stores device pointers

  size_t problem_count = descriptors.size() - end_cudnn_layer;
  // printf("set_cutlass_workspace: problem_count = %d \n", problem_count);

  if (problem_count == 0) {
    return;
  }
  
  std::vector<CutlassConv2dConfig> configs;

  for (int r=0; r < num_rows_to_compute; ++r) {
    for (size_t i=0; i < descriptors.size(); ++i) {
      if (i >= end_cudnn_layer) {
        configs.push_back(descriptors.at(i).cutlass_config);
      }
    }
  }

  assert(problem_count * num_rows_to_compute == configs.size());

  if (quant) {
    cutlass_wgrad_grouped::initialize_problems<int8_t>(configs);
  } 
  else {
    cutlass_wgrad_grouped::initialize_problems<float>(configs); 
  }

  // Allocate device memory to store device pointers

  int64_t device_array_size = (int64_t)sizeof(void*)*problem_count*batch_size;
  tensor_for_device_ptr_A = torch::empty({device_array_size}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
  tensor_for_device_ptr_B = torch::empty({device_array_size}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
  tensor_for_device_ptr_C = torch::empty({device_array_size}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
  tensor_for_device_ptr_D = torch::empty({device_array_size}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
  
  device_ptr_A = (void **)tensor_for_device_ptr_A.data_ptr();
  device_ptr_B = (void **)tensor_for_device_ptr_B.data_ptr();
  device_ptr_C = (void **)tensor_for_device_ptr_C.data_ptr();
  device_ptr_D = (void **)tensor_for_device_ptr_D.data_ptr();

  // if (host_ptr_A != NULL) {free(host_ptr_A);}
  // if (host_ptr_B != NULL) {free(host_ptr_B);}
  // if (host_ptr_C != NULL) {free(host_ptr_C);}
  // if (host_ptr_D != NULL) {free(host_ptr_D);}

  // host_ptr_A = (void **)malloc(sizeof(void*)*problem_count*batch_size);
  // host_ptr_B = (void **)malloc(sizeof(void*)*problem_count*batch_size);
  // host_ptr_C = (void **)malloc(sizeof(void*)*problem_count*batch_size);
  // host_ptr_D = (void **)malloc(sizeof(void*)*problem_count*batch_size);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename dType>
void set_cutlass_device_ptrs(size_t end_cudnn_layer,
                             int _batch_size,
                             std::vector<torch::Tensor>& actvs,
                             std::vector<torch::Tensor>& ograds,
                             std::vector<void *>& wgrads_ptrs,
                             std::vector<void *>& wgrads_ptrs2) {

  assert(actvs.size() == wgrads_ptrs.size());
  assert(ograds.size() == wgrads_ptrs.size());

  assert(device_ptr_A != NULL);
  assert(device_ptr_B != NULL);
  assert(device_ptr_C != NULL);
  assert(device_ptr_D != NULL);

  size_t problem_count = actvs.size() - end_cudnn_layer;
  if (problem_count == 0) {
    return;
  }
  // printf("set_cutlass_device_ptrs: problem_count = %d \n", problem_count);

  size_t batch_size = (size_t)_batch_size;

  host_ptr_A.clear();
  host_ptr_B.clear();
  host_ptr_C.clear();
  host_ptr_D.clear();
  host_ptr_A.resize(batch_size*problem_count);
  host_ptr_B.resize(batch_size*problem_count);
  host_ptr_C.resize(num_rows_to_compute*problem_count);
  host_ptr_D.resize(num_rows_to_compute*problem_count);

  for (size_t problem_idx = 0; problem_idx < problem_count; ++problem_idx) {
    dType * ograds_ptr = (dType *)ograds.at(end_cudnn_layer + problem_idx).data_ptr();
    dType * actvs_ptr = (dType *)actvs.at(end_cudnn_layer + problem_idx).data_ptr();
    size_t ograds_offset = ograds.at(end_cudnn_layer + problem_idx).numel() / batch_size;
    size_t actvs_offset = actvs.at(end_cudnn_layer + problem_idx).numel() / batch_size;

    // printf("%p - %p ( %lld )\n", ograds_ptr, ograds_ptr + ograds.at(end_cudnn_layer + problem_idx).numel(), ograds.at(end_cudnn_layer + problem_idx).numel());
    // printf("%p - %p ( %lld )\n", actvs_ptr, actvs_ptr + actvs.at(end_cudnn_layer + problem_idx).numel(), actvs.at(end_cudnn_layer + problem_idx).numel());

    // std::cout << ograds.at(end_cudnn_layer + problem_idx).is_contiguous() << std::endl;
    // std::cout << actvs.at(end_cudnn_layer + problem_idx).is_contiguous() << std::endl;

    for (size_t example_idx = 0; example_idx < batch_size; ++example_idx) {
      host_ptr_A[example_idx*problem_count + problem_idx] = ograds_ptr;
      host_ptr_B[example_idx*problem_count + problem_idx] = actvs_ptr;
      ograds_ptr += ograds_offset;
      actvs_ptr += actvs_offset;
    }
  }
  // checkCudaErrors(cudaDeviceSynchronize());

  for (int row = 0; row < num_rows_to_compute; ++row) {
    for (size_t problem_idx = 0; problem_idx < problem_count; ++problem_idx) {
      host_ptr_C[row*problem_count + problem_idx] = wgrads_ptrs.at(row*(end_cudnn_layer + problem_count) + end_cudnn_layer + problem_idx);
      host_ptr_D[row*problem_count + problem_idx] = wgrads_ptrs.at(row*(end_cudnn_layer + problem_count) + end_cudnn_layer + problem_idx);
    }
  }

  // checkCudaErrors(cudaDeviceSynchronize());
  // printf("%p %p %lu\n", (void*)device_ptr_A, (void*)&host_ptr_A[0], sizeof(void*)*problem_count*batch_size);
  checkCudaErrors(cudaMemcpyAsync((void*)device_ptr_A, (void*)&host_ptr_A[0], sizeof(void*)*problem_count*batch_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync((void*)device_ptr_B, (void*)&host_ptr_B[0], sizeof(void*)*problem_count*batch_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync((void*)device_ptr_C, (void*)&host_ptr_C[0], sizeof(void*)*problem_count*num_rows_to_compute, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync((void*)device_ptr_D, (void*)&host_ptr_D[0], sizeof(void*)*problem_count*num_rows_to_compute, cudaMemcpyHostToDevice));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_cutlass_best_operation(size_t end_cudnn_layer,
                                int batch_size,
                                std::vector<torch::Tensor>& actvs,
                                std::vector<torch::Tensor>& ograds,
                                bool quant=false,
                                bool test=true) {
  printf("set_cutlass_best_operation start %d %d\n", descriptors.size(), end_cudnn_layer);
  if (descriptors.size() - end_cudnn_layer == 0) {
    return;
  }
  // torch::cuda::synchronize();
  printf("%s\n", cudaGetErrorName(cudaGetLastError()));
  // auto _a = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA, 0));
  printf("Alloc partial_per_example_gradient\n"); 
  auto partial_per_example_gradient = torch::zeros({num_rows_to_compute, (int64_t)partial_per_example_gradient_size}, torch::TensorOptions().device(torch::kCUDA, 0));
  printf("partial_per_example_gradient\n");
  // Set CUTLASS device pointers 
  std::vector<void *> wgrads_ptrs;
  for (int row = 0; row < num_rows_to_compute; ++row) {
    float * wgrads_ptr = (float *)partial_per_example_gradient.index({row}).data_ptr();
    for (size_t i = 0; i < descriptors.size(); ++i) { 
      wgrads_ptrs.push_back(wgrads_ptr); 
      wgrads_ptr += descriptors.at(i).grad_weight_per_example_size; 
      // if (descriptors.at(i).grad_weight_per_example_size % 128 != 0) {
      //   wgrads_ptr += (128 - (descriptors.at(i).grad_weight_per_example_size % 128));
      // }
    }
  }
  printf("set_cutlass_device_ptrs\n");
  if (quant) {
    set_cutlass_device_ptrs<int8_t>(end_cudnn_layer, batch_size, actvs, ograds, wgrads_ptrs, wgrads_ptrs);
  }
  else {
    set_cutlass_device_ptrs<float>(end_cudnn_layer, batch_size, actvs, ograds, wgrads_ptrs, wgrads_ptrs);
  }
  // checkCudaErrors(cudaDeviceSynchronize()); 

  if (test) {
    best_operation_with_workspace = cutlass_wgrad_grouped::get_best_operation(device_ptr_A, device_ptr_B, device_ptr_C, device_ptr_D);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_single_per_example_gradient_cudnn(std::vector<Conv2dDescriptor>& descs,
                                              torch::Tensor &per_example_gradient,
                                              std::vector<torch::Tensor>& actvs,
                                              std::vector<torch::Tensor>& ograds,
                                              size_t end_cudnn_layer,
                                              int example_idx,
                                              bool quant=false) {
  assert(descs.size() == actvs.size());
  assert(descs.size() == ograds.size());

  using namespace torch::indexing;

  // cudnnHandle_t cudnn_handle;
  float alpha = 1.0;
  float beta = 0.0;
  for (int row=0; row < num_rows_to_compute; ++row) {
    int offset = 0;
    float * wgrad_ptr = (float *)per_example_gradient.index({row}).data_ptr();
    for (size_t i=0; i < descs.size(); ++i) {
      Conv2dDescriptor& desc = descs.at(i);

      if (i >= end_cudnn_layer) {
        break;
      }
      // torch::cuda::synchronize();
      // printf("%d %d %d %d\n", actvs.at(i).stride((int64_t)0), actvs.at(i).stride((int64_t)1), actvs.at(i).stride((int64_t)2), actvs.at(i).stride((int64_t)3));
      // printf("%d %d %d %d\n", ograds.at(i).stride((int64_t)0), ograds.at(i).stride((int64_t)1), ograds.at(i).stride((int64_t)2), ograds.at(i).stride((int64_t)3));
      if (quant) {
        checkCudaErrors(cutlass_simt_iwgrad((int8_t *)ograds.at(i).data_ptr() + (example_idx + row) * ograds.at(i).numel() / desc.config.N,
                                            (int8_t *)actvs.at(i).data_ptr() + (example_idx + row) * actvs.at(i).numel() / desc.config.N,
                                            wgrad_ptr,
                                            desc.workspace_ptr,
                                            1,
                                            desc.config.H,
                                            desc.config.W,
                                            desc.config.C,
                                            desc.config.K,
                                            desc.config.R,
                                            desc.config.S,
                                            desc.config.P,
                                            desc.config.Q,
                                            desc.config.pad_h,
                                            desc.config.pad_w,
                                            desc.config.stride_h,
                                            desc.config.stride_w,
                                            desc.config.dilation_h,
                                            desc.config.dilation_w,
                                            1.0,
                                            0.0,
                                            cuda_streams[i % _N_CUDA_STREAMS]));
      }
      else {
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handles[i % _N_CUDA_STREAMS],
                                                  &alpha,
                                                  desc.input_desc,
                                                  (void *)((float *)actvs.at(i).data_ptr() + (example_idx + row) * actvs.at(i).numel() / desc.config.N),
                                                  desc.output_desc,
                                                  (void *)((float *)ograds.at(i).data_ptr() + (example_idx + row) * ograds.at(i).numel() / desc.config.N),
                                                  desc.conv_desc,
                                                  desc.bwd_filter_algo_best,
                                                  desc.workspace_ptr,
                                                  desc.bwd_filter_algo_best_workspace_size,
                                                  &beta,
                                                  desc.filter_desc,
                                                  wgrad_ptr));
      }
      // torch::cuda::synchronize();
      // checkCudaErrors(cudaDeviceSynchronize());
      // float host_ograds[10];
      // checkCudaErrors(cudaMemcpy((void *)host_ograds,
      //                            (void *)((float *)ograds.at(i).data_ptr() + (example_idx + row) * ograds.at(i).numel() / desc.config.N),
      //                            sizeof(float)*10, cudaMemcpyDeviceToHost));
      // std::cout << "Actvs, " << actvs.at(i).flatten().index({Slice(0, 10)}) << std::endl;
      // std::cout << "Ograds, " << ograds.at(i).flatten().index({Slice(0, 10)}) << std::endl;
      // std::cout << "Ograds2, ";
      // for (int _ = 0; _ < 10; ++_) {printf("%f ", host_ograds[_]);}
      // std::cout << std::endl;
      // std::cout << "After cudnn, " << per_example_gradient.index({row}).flatten().index({Slice(0, 10)}) << std::endl;

      wgrad_ptr += desc.grad_weight_per_example_size;
    }
  }

  // std::cout << per_example_gradient.index({0, Slice(0, 64*3*7*7)}) << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_single_per_example_gradient_cutlass(std::vector<Conv2dDescriptor>& descs,
                                                 size_t end_cudnn_layer,
                                                 int batch_size,
                                                 int example_idx) {

  using namespace torch::indexing;

  int problem_count = descs.size() - end_cudnn_layer;

  if (problem_count == 0) {
    return;
  }

  // Update device pointers
  // printf("%p awefafawef \n", best_operation_with_workspace.operation);
  checkCutlass(cutlass_wgrad_grouped::update_ptrs(best_operation_with_workspace,
                                                device_ptr_A + example_idx*problem_count,
                                                device_ptr_B + example_idx*problem_count,
                                                device_ptr_C,
                                                device_ptr_D,
                                                problem_count*num_rows_to_compute));

  // Run cutlass operation
  checkCutlass(cutlass_wgrad_grouped::run(best_operation_with_workspace));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_single_scaling_factor(torch::Tensor& scaling_factors,
                                   int example_idx,
                                   const torch::Tensor partial_per_example_gradient,
                                   std::vector<torch::Tensor>& precomputed_per_example_grad_norms,
                                   float max_norm,
                                   int scale_loss) {
  
  using namespace torch::indexing;

  torch::Tensor norm = torch::frobenius_norm(partial_per_example_gradient, {1}, false);
  compute_scaling_factor_cuda((float *)scaling_factors.index({example_idx}).data_ptr(), (float *)norm.data_ptr(), max_norm, num_rows_to_compute, scale_loss);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void benchmark_cudnn_and_cutlass(std::vector<torch::Tensor>& actvs,
                                std::vector<torch::Tensor>& ograds,
                                size_t end_cudnn_layer,
                                bool quant=false) {
  
  int batch_count = 16;

  printf("%s\n", cudaGetErrorName(cudaGetLastError()));
  // auto _a = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA, 0));
  auto partial_per_example_gradient = torch::zeros({num_rows_to_compute, (int64_t)partial_per_example_gradient_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  // auto partial_per_example_gradient2 = torch::zeros({(int)partial_per_example_gradient_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  printf("partial_per_example_gradient\n");
  
  std::vector<void *> wgrads_ptrs;
  // Set CUTLASS device pointers
  if (descriptors.size() - end_cudnn_layer != 0) {
    for (int row=0 ; row < num_rows_to_compute; ++row) {
      float * wgrads_ptr = (float *)partial_per_example_gradient.index({row}).data_ptr();
      for (size_t i = 0; i < descriptors.size(); ++i) {
        wgrads_ptrs.push_back(wgrads_ptr); 
        wgrads_ptr += descriptors.at(i).grad_weight_per_example_size; 
      }
    }
    printf("set_cutlass_device_ptrs\n");
    if (quant) {
      set_cutlass_device_ptrs<int8_t>(end_cudnn_layer, batch_count, actvs, ograds, wgrads_ptrs, wgrads_ptrs);
    }
    else {
      set_cutlass_device_ptrs<float>(end_cudnn_layer, batch_count, actvs, ograds, wgrads_ptrs, wgrads_ptrs);
    }
    // checkCudaErrors(cudaDeviceSynchronize());
  }
  printf("%s\n", cudaGetErrorName(cudaGetLastError()));
  // torch::cuda::synchronize();
  // checkCudaErrors(cudaDeviceSynchronize());
  // sleep(10); 
  // set_cutlass_best_operation(end_cudnn_layer, batch_count, actvs, ograds, quant, false);
  // auto _ = cutlass_wgrad_grouped::get_best_operation(device_ptr_A, device_ptr_B, device_ptr_C, device_ptr_D);
  for (int example_idx = 0; example_idx < batch_count; example_idx += num_rows_to_compute) {
    // printf("example_idx = %d\n", example_idx);
    // Compute wgrad of front layers using cuDNN
    compute_single_per_example_gradient_cudnn(descriptors, partial_per_example_gradient, actvs, ograds, end_cudnn_layer, example_idx, quant);

    // checkCudaErrors(cudaDeviceSynchronize());

    compute_single_per_example_gradient_cutlass(descriptors, end_cudnn_layer, batch_count, example_idx);

    // checkCudaErrors(cudaDeviceSynchronize());

    torch::cuda::synchronize();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ReturnType clip_and_reduce_grads_conv(std::vector<Conv2dConfig> &configs,
                                                                  std::vector<torch::Tensor>& actvs,
                                                                  std::vector<torch::Tensor>& ograds,
                                                                  std::vector<torch::Tensor>& precomputed_per_example_grads,
                                                                  std::vector<torch::Tensor>& precomputed_per_example_grad_norms,
                                                                  std::vector<torch::Tensor>& linear_actvs,
                                                                  std::vector<torch::Tensor>& linear_ograds,
                                                                  size_t end_non_reweight_layer,
                                                                  size_t end_cudnn_layer,
                                                                  bool loss_reduction_mean = false,
                                                                  int batch_count = 0,
                                                                  float max_norm = 1.0,
                                                                  float noise_multiplier = 1.0,
                                                                  bool quant = false,
                                                                  bool verbose = false,
                                                                  bool time_profile = false,
                                                                  bool memory_profile = false) {

  using namespace torch::indexing;

  auto start_time = std::chrono::high_resolution_clock::now();
  float backward_weight_ms = 0.0;
  float norm_ms = 0.0;
  float clip_reduce_ms = 0.0;
  float add_noise_ms = 0.0;

  // convert to nchw -> nhwc
  for (size_t i = 0; i < actvs.size(); ++i) {
    if (! ograds.at(i).is_contiguous(c10::MemoryFormat::ChannelsLast)) {
      c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromPool());
      // actvs.at(i) = actvs.at(i).contiguous(c10::MemoryFormat::ChannelsLast);
      ograds.at(i) = ograds.at(i).contiguous(c10::MemoryFormat::ChannelsLast);
    }
  }
  c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
  // printf("1 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  auto partial_per_example_gradient = torch::empty({num_rows_to_compute, (int64_t)partial_per_example_gradient_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  // printf("1 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  // Put precomputed_norm at last to compute norm of total gradient
  // To compute clip and reduce efficiently, gather all precomputed grads into a single tensor
  int64_t gathered_per_example_grads_size = 0;
  for (size_t i = 0; i < precomputed_per_example_grads.size(); ++i) {
    gathered_per_example_grads_size += precomputed_per_example_grads.at(i).numel() / batch_count;
  }
  auto gathered_per_example_grads = torch::empty({batch_count, gathered_per_example_grads_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));\
  {
    int64_t offset = 0;
    for (size_t i = 0; i < precomputed_per_example_grads.size(); ++i) {
      gathered_per_example_grads.index_put_({Slice(), Slice(offset, offset + precomputed_per_example_grads.at(i).numel() / batch_count)},
                                            precomputed_per_example_grads.at(i).view({batch_count, -1}));
      offset += precomputed_per_example_grads.at(i).numel() / batch_count;
    }
  }
  gathered_per_example_grads.index_put_({Slice(), gathered_per_example_grads_size}, precomputed_per_example_grad_norms.at(0));
  auto precomputed_per_example_norms = torch::frobenius_norm(gathered_per_example_grads, {1});
  gathered_per_example_grads = gathered_per_example_grads.index({Slice(0, batch_count), Slice(0, gathered_per_example_grads_size)});

  // Set CUTLASS device pointers
  if (descriptors.size() - end_cudnn_layer > 0) {
    std::vector<void *> wgrads_ptrs;
    int64_t offset = 0;
    for (int row = 0; row < num_rows_to_compute; ++row) {
      float* wgrads_ptr = (float *)partial_per_example_gradient.index({row}).data_ptr();
      for (size_t i = 0; i < descriptors.size(); ++i) {
        wgrads_ptrs.push_back(wgrads_ptr);
        wgrads_ptr += descriptors.at(i).grad_weight_per_example_size;
      }
    }
    if (quant) {
      set_cutlass_device_ptrs<int8_t>(end_cudnn_layer, batch_count, actvs, ograds, wgrads_ptrs, wgrads_ptrs);
    }
    else {
      set_cutlass_device_ptrs<float>(end_cudnn_layer, batch_count, actvs, ograds, wgrads_ptrs, wgrads_ptrs);
    }
    // checkCudaErrors(cudaDeviceSynchronize());
  }
  // printf("2 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  // auto partial_per_example_gradient_decoded = torch::zeros({(int)partial_per_example_gradient_size}, torch::TensorOptions().device(torch::kCUDA, 0));
  auto per_batch_gradient = torch::zeros({(int64_t)partial_per_example_gradient_size}, torch::TensorOptions().device(torch::kCUDA, 0));
  auto partial_per_batch_gradient = per_batch_gradient.index({Slice(0, (int64_t)non_reweight_per_example_gradient_size)});

  // Workspace to store scaling factors
  auto scaling_factors = torch::empty({batch_count}, torch::TensorOptions().device(torch::kCUDA, 0));

  LOG_STDERR("Compute per-example gradients and scaling factors", verbose);
  // Compute per-example gradients and scaling factors
  cudaEvent_t events [_N_CUDA_STREAMS];
  for (int i = 0; i < _N_CUDA_STREAMS; ++i) {
    checkCudaErrors(cudaEventCreate(&events[i]));
  }
  for (int example_idx = 0; example_idx < batch_count; example_idx += num_rows_to_compute) {
    // Wait all streams to finish
    // if (example_idx == 0) {
    //   for (size_t i=0; i < _N_CUDA_STREAMS; ++i) {
    //     checkCudaErrors(cudaEventRecord(events[i], NULL));
    //   }
    //   for (size_t i=0; i < _N_CUDA_STREAMS; ++i) {
    //     checkCudaErrors(cudaStreamWaitEvent(cuda_streams[i], events[i], 0));
    //   }
    // }

    // Compute wgrad of front layers using cuDNN
    LOG_STDERR("Compute wgrad of back layers using CUTLASS", verbose);
    // cudaDeviceSynchronize();
    // std::cout << "before partial_per_example_gradient " << partial_per_example_gradient.index({0, Slice(0, 10)}) << std::endl;
    // cudaDeviceSynchronize();
    // Compute wgrad of back layers using CUTLASS
    compute_single_per_example_gradient_cutlass(descriptors, end_cudnn_layer, batch_count, example_idx);

    for (size_t i=0; i < end_cudnn_layer; ++i) {
      checkCudaErrors(cudaEventRecord(events[i], NULL));
    }
    for (size_t i=0; i < end_cudnn_layer; ++i) {
      checkCudaErrors(cudaStreamWaitEvent(cuda_streams[i], events[i], 0));
    }

    compute_single_per_example_gradient_cudnn(descriptors, partial_per_example_gradient, actvs, ograds, end_cudnn_layer, example_idx, quant);

    partial_per_example_gradient.index_put_({Slice(), (int64_t)partial_per_example_gradient_size},
                                            precomputed_per_example_norms.index({Slice(example_idx, example_idx + num_rows_to_compute)}));

    // Wait all streams to finish
    for (size_t i=0; i < end_cudnn_layer; ++i) {
      checkCudaErrors(cudaEventRecord(events[i], cuda_streams[i]));
    }
    for (size_t i=0; i < end_cudnn_layer; ++i) {
      checkCudaErrors(cudaStreamWaitEvent(NULL, events[i], 0));
    }

    // cudaDeviceSynchronize();
    // std::cout << "after partial_per_example_gradient " << partial_per_example_gradient.index({0, Slice(0, 10)}) << std::endl;

    TIME_PROFILE(backward_weight_ms, time_profile);

    // torch::cuda::synchronize();
    LOG_STDERR("Compute scaling factor", verbose);
    // Compute scaling factor
    compute_single_scaling_factor(scaling_factors, example_idx,
                                  partial_per_example_gradient, precomputed_per_example_grad_norms, max_norm, loss_reduction_mean ? batch_count : 1);
    // torch::cuda::synchronize();
    LOG_STDERR("Clip and accumulate", verbose);
    // Clip and accumulate
    // per_batch_gradient += partial_per_example_gradient.index({Slice(0, (int)non_reweight_per_example_gradient_size)}) * scaling_factors.index({example_idx});
    if (non_reweight_per_example_gradient_size > 0) {
      // checkCUBLAS(cublasSaxpy(cublas_handles[0],
      //                         (int)non_reweight_per_example_gradient_size,
      //                         (float *)scaling_factors.index({example_idx}).data_ptr(),
      //                         (float *)partial_per_example_gradient.data_ptr(),
      //                         1,
      //                         (float *)per_batch_gradient.data_ptr(),
      //                         1));
      if (num_rows_to_compute > 1) {
        auto non_reweight_partial_per_example = partial_per_example_gradient.index({Slice(), Slice(0, (int)non_reweight_per_example_gradient_size)});
        non_reweight_partial_per_example.mul_(scaling_factors.index({Slice(example_idx, example_idx + num_rows_to_compute)}).view({num_rows_to_compute, 1}));
        partial_per_batch_gradient.add_(non_reweight_partial_per_example.sum({0}));
      }
      else {
        partial_per_batch_gradient.add_(partial_per_example_gradient.index({0, Slice(0, (int64_t)non_reweight_per_example_gradient_size)}), scaling_factors.index({example_idx}).item());
      }

      // torch::cuda::synchronize();
      // std::cout << "partial_per_batch" << partial_per_batch_gradient.index({Slice(0, 10)}) << std::endl;
      // std::cout << "per_batch" <<  per_batch_gradient.index({Slice(0, 10)}) << std::endl;
    }


    // for (size_t i=0; i < _N_CUDA_STREAMS; ++i) {
    //   checkCudaErrors(cudaEventRecord(events[i], NULL));
    // }
    // for (size_t i=0; i < _N_CUDA_STREAMS; ++i) {
    //   checkCudaErrors(cudaStreamWaitEvent(cuda_streams[i], events[i], 0));
    // }

    // torch::cuda::synchronize();
    TIME_PROFILE(clip_reduce_ms, time_profile);
  }
  // printf("Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  // Scale and reduce pre-computed per-example grads
  LOG_STDERR("Scale and reduce pre-computed per-example grads", verbose);
  int64_t per_batch_gradient_from_precomputed_size = 0;
  std::vector<int64_t> per_batch_gradient_sizes_from_precomputed;
  for (auto grad : precomputed_per_example_grads) {
    int64_t gradient_size = grad.numel();
    per_batch_gradient_from_precomputed_size += gradient_size / batch_count;
    per_batch_gradient_sizes_from_precomputed.push_back(gradient_size / batch_count);
  }
  gathered_per_example_grads.mul_(scaling_factors.view({batch_count, 1}));
  auto per_batch_gradient_from_precomputed = torch::sum(gathered_per_example_grads, {0});
  TIME_PROFILE(clip_reduce_ms, time_profile);
  add_noise(per_batch_gradient_from_precomputed, max_norm*noise_multiplier);
  // per_batch_gradient_from_precomputed.add_(torch::normal(0.0, max_norm*noise_multiplier, per_batch_gradient_from_precomputed.sizes(), c10::nullopt, torch::TensorOptions().device(torch::kCUDA, 0)));
  TIME_PROFILE(add_noise_ms, time_profile);
  // printf("Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);

  std::vector<torch::Tensor> per_batch_grads_from_precomputed;
  {
    int offset = 0;
    for (int grad_idx = 0; grad_idx < precomputed_per_example_grads.size(); ++grad_idx) {
      auto& grad = precomputed_per_example_grads.at(grad_idx);

      std::vector<int64_t> grad_shape;
      for (size_t i = 1; i < grad.sizes().size(); ++i){
        grad_shape.push_back(grad.sizes()[i]);
      }
      
      per_batch_grads_from_precomputed.push_back(per_batch_gradient_from_precomputed.index({Slice(offset, offset + per_batch_gradient_sizes_from_precomputed.at(grad_idx))}).view(grad_shape));
      
      offset += per_batch_gradient_sizes_from_precomputed.at(grad_idx);
    }
  }

  std::vector<torch::Tensor> per_batch_grads;
  // Split finished per-batch gradients and add to list
  LOG_STDERR("Split finished per-batch gradients and add to list", verbose);
  int64_t offset = 0;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    if (i >= end_non_reweight_layer){
      per_batch_grads.push_back(per_batch_gradient.index({Slice(offset, offset + descriptors.at(i).grad_weight_per_example_size)}).view(descriptors.at(i).filter_shape));
    }
    else{
      per_batch_grads.push_back(per_batch_gradient.index({Slice(offset, offset + descriptors.at(i).grad_weight_per_example_size)}));
    }
    offset += descriptors.at(i).grad_weight_per_example_size;
  }
  assert(per_batch_grads.size() == descriptors.size());

  // Scale output grads
  LOG_STDERR("Scale output grads for reweight", verbose);
  for (size_t i = 0; i < descriptors.size(); ++i) {
    if (i < end_non_reweight_layer){
      continue;
    }

    std::vector<int64_t> scaling_factors_shape;
    scaling_factors_shape.push_back(scaling_factors.size(0));
    for (size_t j = 0; j < actvs.at(i).sizes().size() - 1; ++j){
      scaling_factors_shape.push_back(1);
    }

    c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromExternal(cuda_streams[i%_N_CUDA_STREAMS], 0));

    torch::Tensor temp_actv;
    torch::Tensor temp_ograd;

    if (quant) {
      temp_actv = actvs.at(i).to(torch::TensorOptions().dtype(torch::kFloat));
      temp_ograd = torch::mul(ograds.at(i), scaling_factors.view(scaling_factors_shape));
    }
    else {
      // torch::mul_out(ograds.at(i), ograds.at(i), scaling_factors.view(scaling_factors_shape));
      // temp_ograd = torch::mul(ograds.at(i), scaling_factors.view(scaling_factors_shape));
      temp_actv = actvs.at(i).contiguous(c10::MemoryFormat::Contiguous);
      temp_ograd = torch::mul(ograds.at(i), scaling_factors.view(scaling_factors_shape)).contiguous(c10::MemoryFormat::Contiguous);
    }

    LOG_STDERR("Compute per-batch gradient for rewight layers", verbose);
    float alpha = 1.0;
    float beta = 0.0;

    // Compute per-batch gradient for rewight layers
    Conv2dDescriptor& descriptor = descriptors.at(i);

    if (quant) {
      checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handles[i % _N_CUDA_STREAMS],
                                                &alpha,
                                                descriptor.input_batch_desc,
                                                temp_actv.data_ptr(),
                                                descriptor.output_batch_desc,
                                                temp_ograd.data_ptr(),
                                                descriptor.conv_desc,
                                                descriptor.bwd_filter_batch_algo_best,
                                                descriptor.batch_workspace_ptr,
                                                descriptor.bwd_filter_batch_algo_best_workspace_size,
                                                &beta,
                                                descriptor.filter_desc,
                                                per_batch_grads.at(i).data_ptr()));
    }
    else {
      checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handles[i % _N_CUDA_STREAMS],
                                                &alpha,
                                                descriptor.input_batch_desc,
                                                temp_actv.data_ptr(),
                                                descriptor.output_batch_desc,
                                                temp_ograd.data_ptr(),
                                                descriptor.conv_desc,
                                                descriptor.bwd_filter_batch_algo_best,
                                                descriptor.batch_workspace_ptr,
                                                descriptor.bwd_filter_batch_algo_best_workspace_size,
                                                &beta,
                                                descriptor.filter_batch_desc,
                                                per_batch_grads.at(i).data_ptr()));
    }
    per_batch_grads.at(i) = per_batch_grads.at(i).contiguous(c10::MemoryFormat::ChannelsLast);
  }
  c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
  // Wait all streams to finish
  for (size_t i=0; i < _N_CUDA_STREAMS; ++i) {
    checkCudaErrors(cudaEventRecord(events[i], cuda_streams[i]));
  }
  for (size_t i=0; i < _N_CUDA_STREAMS; ++i) {
    checkCudaErrors(cudaStreamWaitEvent(NULL, events[i], 0));
  }
  // cudaDeviceSynchronize();
  // std::cout << "Scaling factors " << scaling_factors << std::endl;
  // std::cout << "After reweight, per_batch" <<  per_batch_grads.at(0).index({Slice(0, 10)}) << std::endl;

  TIME_PROFILE(clip_reduce_ms, time_profile);
  // per_batch_gradient.add_(torch::normal(0.0, max_norm*noise_multiplier, per_batch_gradient.sizes(), c10::nullopt, torch::TensorOptions().device(torch::kCUDA, 0)));
  add_noise(per_batch_gradient, max_norm*noise_multiplier);
  TIME_PROFILE(add_noise_ms, time_profile);

  LOG_STDERR("Scale output grads (for linear layer) for reweight", verbose);
  std::vector<torch::Tensor> scaled_linear_ograds;
  for (size_t i = 0; i < linear_actvs.size(); ++i) {
    std::vector<int64_t> scaling_factors_shape;
    scaling_factors_shape.push_back(scaling_factors.size(0));
    for (size_t j = 0; j < linear_actvs.at(i).sizes().size() - 1; ++j){
      scaling_factors_shape.push_back(1);
    }
    // torch::mul_out(linear_actvs.at(i), linear_actvs.at(i), scaling_factors.view(scaling_factors_shape));
    // torch::mul_out(linear_ograds.at(i), linear_ograds.at(i), scaling_factors.view(scaling_factors_shape));
    scaled_linear_ograds.push_back(torch::mul(linear_ograds.at(i), scaling_factors.view(scaling_factors_shape)));
  }

  std::vector<torch::Tensor> per_batch_linear_grads;
  // Split finished per-batch gradients and add to list
  LOG_STDERR("Split finished per-batch gradients for linear and add to list", verbose);
  for (size_t i = 0; i < linear_actvs.size(); ++i) {
    per_batch_linear_grads.push_back(torch::empty({linear_ograds.at(i).sizes()[1], linear_actvs.at(i).sizes()[1]}, torch::TensorOptions().device(torch::kCUDA, 0)));
  }
  
  for (size_t i = 0; i < linear_actvs.size(); ++i) {
    // std::cout << "Linear actvs, " << linear_actvs.at(i).flatten().index({Slice(0, 10)}) << std::endl;
    // std::cout << "Linear ograds, " << linear_ograds.at(i).flatten().index({Slice(0, 10)}) << std::endl;
    torch::matmul_out(per_batch_linear_grads.at(i), scaled_linear_ograds.at(i).transpose(0, 1), linear_actvs.at(i));
    TIME_PROFILE(clip_reduce_ms, time_profile);
    // per_batch_linear_grads.at(i).add_(torch::normal(0.0, max_norm*noise_multiplier, per_batch_linear_grads.at(i).sizes(), c10::nullopt, torch::TensorOptions().device(torch::kCUDA, 0)));
    add_noise(per_batch_linear_grads.at(i), max_norm*noise_multiplier);
    TIME_PROFILE(add_noise_ms, time_profile);
  }

  TIME_PROFILE(clip_reduce_ms, time_profile);

  return ReturnType({per_batch_grads, per_batch_grads_from_precomputed, per_batch_linear_grads}, backward_weight_ms, norm_ms, clip_reduce_ms, add_noise_ms, total_ws_size + total_per_batch_cudnn_workspace);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ReturnType get_clip_and_reduced_grads_conv(std::vector<Conv2dConfig> &configs,
                                          std::vector<torch::Tensor>& actvs,
                                          std::vector<torch::Tensor>& ograds,
                                          std::vector<torch::Tensor>& precomputed_per_example_grads,
                                          std::vector<torch::Tensor>& precomputed_per_example_grad_norms,
                                          std::vector<torch::Tensor>& linear_actvs,
                                          std::vector<torch::Tensor>& linear_ograds,
                                          bool loss_reduction_mean,
                                          int batch_count,
                                          float max_norm,
                                          float noise_multiplier,
                                          bool quant,
                                          bool verbose,
                                          bool profile_time,
                                          bool profile_memory) {
  assert(configs.size() == actvs.size());
  assert(configs.size() == ograds.size());
  assert(linear_ograds.size() == linear_actvs.size());

  using namespace torch::indexing;

  if (_first_run) {

    LOG_STDERR("Set descriptor conv (first run)", verbose);
    set_descriptors_conv(configs, quant);
    LOG_STDERR("Find best algorithm for per-batch weight gradient during first run", verbose);
    set_cudnn_per_batch_algorithm(actvs, ograds);
    
    ///////////////////////////////////////////////////////////////////////////////////

    // Profile to find best end-cudnn-layer

    // Set per-batch algo workspace
    for (size_t i = 0; i < descriptors.size(); ++i) { 
      if (descriptors.at(i).bwd_filter_batch_algo_best_workspace_size > max_bwd_filter_batch_algo_best_workspace_size) {
        max_bwd_filter_batch_algo_best_workspace_size = descriptors.at(i).bwd_filter_batch_algo_best_workspace_size;
      }
    }

    if (max_bwd_filter_batch_algo_best_workspace_size > 0) {
      tensor_for_bwd_filter_batch_workspace = torch::empty({(int64_t)max_bwd_filter_batch_algo_best_workspace_size}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
      bwd_filter_batch_workspace = (void *)tensor_for_bwd_filter_batch_workspace.data_ptr();
    }
    else {
      tensor_for_bwd_filter_batch_workspace = torch::empty({0}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
    }

    auto min_runtime_us = std::chrono::microseconds(1000000000);
    auto prev_runtime_us = std::chrono::microseconds(1000000000);
    auto increase_count = 0;
    for (size_t end_cudnn_layer = (quant? 1 : 0); end_cudnn_layer < configs.size() + 1; ++end_cudnn_layer) { // FIXME
      // Initialize cutlass_wgrad_grouped library
      if (quant) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

          // cutlass_wgrad_grouped::initialize_int_tensorop(); 
#if defined(_USE_TENSOR_CORE)
          cutlass_wgrad_grouped::initialize_int_tensorop();
#else
          cutlass_wgrad_grouped::initialize_int();
#endif
      }
      else {
        cutlass_wgrad_grouped::initialize_float(); 
      }

      LOG_STDERR("Set cudnn workspace and compute non-reweight per-example gradient size", verbose);
      // Set cudnn workspace and compute non-reweight per-example gradient size
      set_cudnn_workspace(end_cudnn_layer); 

      LOG_STDERR("Set cutlass workspace", verbose);
      // Set cutlass workspace
      set_cutlass_workspace(end_cudnn_layer, batch_count, quant);

      // printf("Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
      LOG_STDERR("Set cutlass best operation", verbose);
      // Set cutlass best operation
      set_cutlass_best_operation(end_cudnn_layer, batch_count, actvs, ograds, quant);
      if (best_operation_with_workspace.operation == NULL && end_cudnn_layer < configs.size()) {
        {
          std::ostringstream stringStream;
          stringStream << "Current end cudnn layer = " << end_cudnn_layer << ", Failed in CUTLASS";
          std::string copyOfStr = stringStream.str();
          LOG_STDERR(copyOfStr, true);
        }
        cutlass_wgrad_grouped::finalize();
        continue;
      }

      // Warm up
      for (int i = 0; i < 3; ++i) {
        benchmark_cudnn_and_cutlass(actvs, ograds, end_cudnn_layer, quant); 
      }
      // printf("Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
      checkCudaErrors(cudaDeviceSynchronize());
      auto startTime = std::chrono::high_resolution_clock::now();

      // Run clip_and_reduce_grads
      for (int i = 0; i < 5; ++i) {
        benchmark_cudnn_and_cutlass(actvs, ograds, end_cudnn_layer, quant);
      }

      checkCudaErrors(cudaDeviceSynchronize());
      auto endTime = std::chrono::high_resolution_clock::now();

      cutlass_wgrad_grouped::finalize();

      auto runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
      if (runtime_us < min_runtime_us) {
        best_end_cudnn_layer = end_cudnn_layer;
        min_runtime_us = runtime_us;
      }
      if (runtime_us <= prev_runtime_us) {
        increase_count = std::max(0, increase_count - 1);
      }
      else {
        increase_count += 1;
      }
      if (increase_count == THRESHOLD_INCREASE_COUNT_NON_CUDNN) {
        break;
      }
      prev_runtime_us = runtime_us;

      std::ostringstream stringStream;
      stringStream << "Current end cudnn layer = " << end_cudnn_layer << ", runtime = " << runtime_us.count() << " us";
      std::string copyOfStr = stringStream.str();
      LOG_STDERR(copyOfStr, true);
    }

    // best_end_cudnn_layer = 1; // FIXME
    {
      std::ostringstream stringStream;
      stringStream << "Best end-cudnn-layer = " << best_end_cudnn_layer;
      std::string copyOfStr = stringStream.str();
      LOG_STDERR(copyOfStr, true);
    }
    set_cudnn_workspace(best_end_cudnn_layer); 

    // Initialize cutlass_wgrad_grouped library
    if (quant) {
#if defined(_USE_TENSOR_CORE)
      cutlass_wgrad_grouped::initialize_int_tensorop();
#else
      cutlass_wgrad_grouped::initialize_int();
#endif
    }
    else {
      cutlass_wgrad_grouped::initialize_float(); 
    }
    // printf("a Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
    // Set cutlass workspace with best end-cudnn-layer
    set_cutlass_workspace(best_end_cudnn_layer, batch_count, quant);

    // Set cutlass best operation with best end-cudnn-layer
    set_cutlass_best_operation(best_end_cudnn_layer, batch_count, actvs, ograds, quant);

    ///////////////////////////////////////////////////////////////////////////////////
    
    // Profile to find best end-non-reweight-layer

    LOG_STDERR("Profile...", verbose);

    min_runtime_us = std::chrono::microseconds(1000000000);
    prev_runtime_us = std::chrono::microseconds(1000000000);
    increase_count = 0;
    // printf("b Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
    // Profile to find best end-non-reweight-layer
    for (size_t end_non_reweight_layer = 0; end_non_reweight_layer < configs.size() + 1; ++end_non_reweight_layer) { // FIXME

      // Compute non-reweight per-example gradient size
      non_reweight_per_example_gradient_size = 0;
      for (size_t i = 0; i < configs.size(); ++i) {
        if (i < end_non_reweight_layer) {
          non_reweight_per_example_gradient_size += (size_t)configs.at(i).K*configs.at(i).C*configs.at(i).R*configs.at(i).S;
        }
      }
      set_per_batch_cudnn_workspace(end_non_reweight_layer);
      // printf("c Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);

      // Warm up
      for (int i = 0; i < 1; ++i) {
        auto _ = clip_and_reduce_grads_conv(configs,
                                            actvs,
                                            ograds,
                                            precomputed_per_example_grads,
                                            precomputed_per_example_grad_norms,
                                            linear_actvs,
                                            linear_ograds,
                                            end_non_reweight_layer,
                                            best_end_cudnn_layer,
                                            loss_reduction_mean,
                                            batch_count,
                                            max_norm,
                                            noise_multiplier,
                                            quant,
                                            false,
                                            false,
                                            false); 
      }

      checkCudaErrors(cudaDeviceSynchronize()); 
      auto startTime = std::chrono::high_resolution_clock::now();

      // Run clip_and_reduce_grads
      for (int i = 0; i < 3; ++i) {
        auto _ = clip_and_reduce_grads_conv(configs,
                                            actvs,
                                            ograds,
                                            precomputed_per_example_grads,
                                            precomputed_per_example_grad_norms,
                                            linear_actvs,
                                            linear_ograds,
                                            end_non_reweight_layer,
                                            best_end_cudnn_layer,
                                            loss_reduction_mean,
                                            batch_count,
                                            max_norm,
                                            noise_multiplier,
                                            quant,
                                            false,
                                            false,
                                            false);
      }

      checkCudaErrors(cudaDeviceSynchronize());
      auto endTime = std::chrono::high_resolution_clock::now();

      auto runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
      if (runtime_us < min_runtime_us) {
        best_end_non_reweight_layer = end_non_reweight_layer;
        min_runtime_us = runtime_us;
      }
      if (runtime_us <= prev_runtime_us) {
        increase_count = std::max(0, increase_count - 1);
      }
      else {
        increase_count += 1;
      }
      if (increase_count == THRESHOLD_INCREASE_COUNT_NON_REWEIGHT) {
        break;
      }
      prev_runtime_us = runtime_us;

      std::ostringstream stringStream;
      stringStream << "Current end non-reweight layer = " << end_non_reweight_layer << ", runtime = " << runtime_us.count() << " us";
      std::string copyOfStr = stringStream.str();
      LOG_STDERR(copyOfStr, true);
    }

    // best_end_non_reweight_layer = 2; // FIXME
    std::ostringstream stringStream;
    stringStream << "Best end-non-reweight-layer = " << best_end_non_reweight_layer;
    std::string copyOfStr = stringStream.str();
    LOG_STDERR(copyOfStr, true);

    // Find maximum workspace among reweight layers
    for (size_t i = 0; i < descriptors.size(); ++i) { 
      if (i < best_end_non_reweight_layer) {
        continue;
      }
      if (descriptors.at(i).bwd_filter_batch_algo_best_workspace_size > max_bwd_filter_batch_algo_best_workspace_size) {
        max_bwd_filter_batch_algo_best_workspace_size = descriptors.at(i).bwd_filter_batch_algo_best_workspace_size;
      }
    }

    // Allocate per-batch algo workspace
    if (max_bwd_filter_batch_algo_best_workspace_size > 0) {
      tensor_for_bwd_filter_batch_workspace = torch::empty({(int64_t)max_bwd_filter_batch_algo_best_workspace_size}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
      bwd_filter_batch_workspace = (void *)tensor_for_bwd_filter_batch_workspace.data_ptr();
    }
    else {
      tensor_for_bwd_filter_batch_workspace = torch::empty({0}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
    }

    // Compute non-reweight per-example gradient size
    non_reweight_per_example_gradient_size = 0;
    for (size_t i = 0; i < configs.size(); ++i) {
      if (i < best_end_non_reweight_layer) {
        non_reweight_per_example_gradient_size += (size_t)configs.at(i).K*configs.at(i).C*configs.at(i).R*configs.at(i).S;
      }
    }
    set_per_batch_cudnn_workspace(best_end_non_reweight_layer);
  }

  _first_run = false;

  // auto _ = clip_and_reduce_grads_conv(configs,
  //                                   actvs,
  //                                   ograds,
  //                                   precomputed_per_example_grads,
  //                                   precomputed_per_example_grad_norms,
  //                                   linear_actvs,
  //                                   linear_ograds,
  //                                   best_end_non_reweight_layer,
  //                                   best_end_cudnn_layer,
  //                                   batch_count,
  //                                   max_norm,
  //                                   noise_multiplier,
  //                                   quant,
  //                                   verbose,
  //                                   profile_time,
  //                                   profile_memory);

  return clip_and_reduce_grads_conv(configs,
                                    actvs,
                                    ograds,
                                    precomputed_per_example_grads,
                                    precomputed_per_example_grad_norms,
                                    linear_actvs,
                                    linear_ograds,
                                    best_end_non_reweight_layer,
                                    best_end_cudnn_layer,
                                    loss_reduction_mean,
                                    batch_count,
                                    max_norm,
                                    noise_multiplier,
                                    quant,
                                    verbose,
                                    profile_time,
                                    profile_memory);
}