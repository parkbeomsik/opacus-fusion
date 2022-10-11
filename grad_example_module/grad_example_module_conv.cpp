#include <torch/extension.h>
#include "ATen/cudnn/Handles.h"
#include <c10/cuda/CUDAStream.h>

#include <vector>
#include <chrono>
#include <algorithm>

#include "structure.h"
#include "error_helper.h"

#include "cutlass_wgrad_grouped.h"

#include "compute_scaling_factor_cuda.h"

#include "grad_example_module_conv.h"
#include "utils.h"

#define THRESHOLD_INCREASE_COUNT 4

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global variables

bool _first_run = true;

std::vector<Conv2dDescriptor> descriptors;

size_t partial_per_example_gradient_size = 0;
size_t non_reweight_per_example_gradient_size = 0;
size_t best_end_non_reweight_layer = 3;
size_t best_end_cudnn_layer = 3;

size_t max_bwd_filter_batch_algo_best_workspace_size = 0;
torch::Tensor tensor_for_bwd_filter_batch_workspace;
void * bwd_filter_batch_workspace = NULL;

#define _N_CUDA_STREAMS 100
cudaStream_t cuda_streams[_N_CUDA_STREAMS];
cudnnHandle_t cudnn_handles[_N_CUDA_STREAMS];

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

cutlass_wgrad_grouped::OperationWithWorkspace best_operation_with_workspace;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_descriptors_conv(std::vector<Conv2dConfig> &configs, bool quant=false, int end_non_reweight_layer=3) {
  // Create custom cuda streams
  for (int i=0; i < _N_CUDA_STREAMS; ++i) {
    checkCudaErrors(cudaStreamCreate(&cuda_streams[i]));
    checkCUDNN(cudnnCreate(&cudnn_handles[i]));
    checkCUDNN(cudnnSetStream(cudnn_handles[i], cuda_streams[i]));
  }

  descriptors.clear();
  descriptors.resize(configs.size());

  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;

  size_t total_ws_size = 0;
  // this->batch_size = trainable_layer_list.at(0)->get_input_shape()[0];
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

    checkCUDNN(cudnnCreateTensorDescriptor(&descriptors.at(i).input_batch_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(descriptors.at(i).input_batch_desc,
                                        CUDNN_TENSOR_NHWC,
                                        dtype,
                                        config->N,
                                        config->C,
                                        config->H,
                                        config->W));

    checkCUDNN(cudnnCreateTensorDescriptor(&descriptors.at(i).output_batch_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(descriptors.at(i).output_batch_desc,
                                        CUDNN_TENSOR_NHWC,
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
    checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(at::native::getCudnnHandle(),
                                                            descriptors.at(i).input_desc,
                                                            temp_x.data_ptr(),
                                                            descriptors.at(i).output_desc,
                                                            temp_y.data_ptr(),
                                                            descriptors.at(i).conv_desc,
                                                            descriptors.at(i).filter_desc,
                                                            temp_w.data_ptr(),
                                                            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                                                            &returned_algo_count,
                                                            bwd_filter_algo_perf,
                                                            temp_ws.data_ptr(),
                                                            temp_ws_size));
    descriptors.at(i).bwd_filter_algo_best = bwd_filter_algo_perf[0].algo;
    descriptors.at(i).bwd_filter_algo_best_workspace_size = bwd_filter_algo_perf[0].memory;
    
    total_ws_size += bwd_filter_algo_perf[0].memory;
    descriptors.at(i).grad_weight_per_example_size_in_bytes = (size_t)sizeof(float)\
                                                                *config->K\
                                                                *config->C\
                                                                *config->R\
                                                                *config->S;
    descriptors.at(i).grad_weight_per_example_size = (size_t)config->K*config->C*config->R*config->S;
    descriptors.at(i).workspace_size_in_bytes = bwd_filter_algo_perf[0].memory;
    total_ws_size += bwd_filter_algo_perf[0].memory;
    partial_per_example_gradient_size += (size_t)config->K*config->C*config->R*config->S;

    descriptors.at(i).filter_shape = {config->K, config->C, config->R, config->S};


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
    int temp_ws_size = (size_t)sizeof(float)*config.N*config.C*config.H*config.W;
    torch::Tensor temp_ws = torch::empty({config.N*config.C*config.H*config.W}, torch::TensorOptions().device(torch::kCUDA, 0));
    torch::Tensor wgrad = torch::empty({config.K*config.C*config.R*config.S}, torch::TensorOptions().device(torch::kCUDA, 0));

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo_perf[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(at::native::getCudnnHandle(),
                                                            descriptors.at(i).input_batch_desc,
                                                            actvs.at(i).data_ptr(),
                                                            descriptors.at(i).output_batch_desc,
                                                            ograds.at(i).data_ptr(),
                                                            descriptors.at(i).conv_desc,
                                                            descriptors.at(i).filter_desc,
                                                            wgrad.data_ptr(),
                                                            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                                                            &returned_algo_count,
                                                            bwd_filter_algo_perf,
                                                            temp_ws.data_ptr(),
                                                            temp_ws_size));
    descriptors.at(i).bwd_filter_batch_algo_best = bwd_filter_algo_perf[0].algo;
    descriptors.at(i).bwd_filter_batch_algo_best_workspace_size = bwd_filter_algo_perf[0].memory;

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

void set_cutlass_workspace(size_t end_cudnn_layer, int batch_size) {
  // Set device workspace which stores device pointers

  size_t problem_count = descriptors.size() - end_cudnn_layer;
  // printf("set_cutlass_workspace: problem_count = %d \n", problem_count);

  if (problem_count == 0) {
    return;
  }
  
  std::vector<CutlassConv2dConfig> configs;

  for (size_t i=0; i < descriptors.size(); ++i) {
    if (i >= end_cudnn_layer) {
      configs.push_back(descriptors.at(i).cutlass_config);
    }
  }

  cutlass_wgrad_grouped::initialize_problems(configs);

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

  std::vector<void *> host_ptr_A;
  std::vector<void *> host_ptr_B;
  std::vector<void *> host_ptr_C;
  std::vector<void *> host_ptr_D;
  host_ptr_A.resize(batch_size*problem_count);
  host_ptr_B.resize(batch_size*problem_count);
  host_ptr_C.resize(problem_count);
  host_ptr_D.resize(problem_count);


  for (size_t example_idx = 0; example_idx < batch_size; ++example_idx) {
    for (size_t problem_idx = 0; problem_idx < problem_count; ++problem_idx) {
      host_ptr_A[example_idx*problem_count + problem_idx] = ograds.at(end_cudnn_layer + problem_idx).index({(int64_t)example_idx}).data_ptr();
      host_ptr_B[example_idx*problem_count + problem_idx] = actvs.at(end_cudnn_layer + problem_idx).index({(int64_t)example_idx}).data_ptr();
    }
  }

  for (size_t problem_idx = 0; problem_idx < problem_count; ++problem_idx) {
    host_ptr_C[problem_idx] = wgrads_ptrs2.at(end_cudnn_layer + problem_idx);
    host_ptr_D[problem_idx] = wgrads_ptrs.at(end_cudnn_layer + problem_idx);
  }

  checkCudaErrors(cudaMemcpyAsync((void*)device_ptr_A, (void*)&host_ptr_A[0], sizeof(void*)*problem_count*batch_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync((void*)device_ptr_B, (void*)&host_ptr_B[0], sizeof(void*)*problem_count*batch_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync((void*)device_ptr_C, (void*)&host_ptr_C[0], sizeof(void*)*problem_count, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync((void*)device_ptr_D, (void*)&host_ptr_D[0], sizeof(void*)*problem_count, cudaMemcpyHostToDevice));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_cutlass_best_operation(size_t end_cudnn_layer,
                                int batch_size,
                                std::vector<torch::Tensor>& actvs,
                                std::vector<torch::Tensor>& ograds) {

  if (descriptors.size() - end_cudnn_layer == 0) {
    return;
  }

  auto partial_per_example_gradient = torch::zeros({(int)partial_per_example_gradient_size}, torch::TensorOptions().device(torch::kCUDA, 0));
  auto partial_per_example_gradient2 = torch::zeros({(int)partial_per_example_gradient_size}, torch::TensorOptions().device(torch::kCUDA, 0));

  // Set CUTLASS device pointers
  std::vector<void *> wgrads_ptrs;
  int64_t offset = 0;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    wgrads_ptrs.push_back(partial_per_example_gradient.index({offset}).data_ptr());
    offset += descriptors.at(i).grad_weight_per_example_size;
  }
  std::vector<void *> wgrads_ptrs2;
  offset = 0;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    wgrads_ptrs2.push_back(partial_per_example_gradient2.index({offset}).data_ptr());
    offset += descriptors.at(i).grad_weight_per_example_size;
  }

  set_cutlass_device_ptrs(end_cudnn_layer, batch_size, actvs, ograds, wgrads_ptrs, wgrads_ptrs2);

  best_operation_with_workspace = cutlass_wgrad_grouped::get_best_operation(device_ptr_A, device_ptr_B, device_ptr_C, device_ptr_D);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_single_per_example_gradient_cudnn(std::vector<Conv2dDescriptor>& descs,
                                              torch::Tensor &per_example_gradient,
                                              std::vector<torch::Tensor>& actvs,
                                              std::vector<torch::Tensor>& ograds,
                                              size_t end_cudnn_layer,
                                              int example_idx) {
  assert(descs.size() == actvs.size());
  assert(descs.size() == ograds.size());

  using namespace torch::indexing;

  // cudnnHandle_t cudnn_handle;
  int alpha = 1;
  int beta = 0;
  int offset = 0;
  for (size_t i=0; i < descs.size(); ++i) {
    if (i >= end_cudnn_layer) {
      return;
    }
    Conv2dDescriptor& desc = descs.at(i);

    checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handles[i % _N_CUDA_STREAMS],
                                              &alpha,
                                              desc.input_desc,
                                              actvs.at(i).index({example_idx, Slice(), Slice(), Slice()}).data_ptr(),
                                              desc.output_desc,
                                              ograds.at(i).index({example_idx, Slice(), Slice(), Slice()}).data_ptr(),
                                              desc.conv_desc,
                                              desc.bwd_filter_algo_best,
                                              desc.workspace_ptr,
                                              desc.bwd_filter_algo_best_workspace_size,
                                              &beta,
                                              desc.filter_desc,
                                              per_example_gradient.index({Slice(offset, offset + desc.grad_weight_per_example_size)}).data_ptr()));
    
    offset += desc.grad_weight_per_example_size;
  }
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
                                                problem_count));

  // Run cutlass operation
  checkCutlass(cutlass_wgrad_grouped::run(best_operation_with_workspace));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_single_scaling_factor(torch::Tensor& scaling_factors,
                                   int example_idx,
                                   torch::Tensor& partial_sums,
                                   const torch::Tensor partial_per_example_gradient,
                                   std::vector<torch::Tensor>& precomputed_per_example_grad_norms,
                                   float max_norm) {
  
  using namespace torch::indexing;
  // for (int i=0; i < partial_sums.sizes()[0]; ++i) {
  //   if (i == 0) {
      // torch::Tensor partial_sum = partial_sums.index({0});
      // torch::frobenius_norm_out(partial_sum, partial_per_example_gradient, {0}, false);
  //   }
  //   else {
  //     torch::Tensor partial_sum = partial_sums.index({i});
  //     torch::frobenius_norm_out(partial_sum, precomputed_per_example_grad_norms.at(i - 1).index({example_idx}), {0}, false);
  //   }
  // }

  torch::Tensor norm = torch::frobenius_norm(partial_per_example_gradient, {0}, false);

  compute_scaling_factor_cuda((float *)scaling_factors.index({example_idx}).data_ptr(), (float *)norm.data_ptr(), max_norm);
  // checkCudaErrors(cudaDeviceSynchronize());
  // torch::cuda::synchronize();

  // scaling_factors.index_put_({example_idx}, (max_norm / (norm + 1e-6)).clamp(0.0, 1.0));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void benchmark_cudnn_and_cutlass(std::vector<torch::Tensor>& actvs,
                                std::vector<torch::Tensor>& ograds,
                                size_t end_cudnn_layer) {
  
  int batch_count = 20;

  auto partial_per_example_gradient = torch::zeros({(int)partial_per_example_gradient_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  auto partial_per_example_gradient2 = torch::zeros({(int)partial_per_example_gradient_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  
  // Set CUTLASS device pointers
  if (descriptors.size() - end_cudnn_layer > 0) {
    std::vector<void *> wgrads_ptrs;
    int64_t offset = 0;
    for (size_t i = 0; i < descriptors.size(); ++i) {
      wgrads_ptrs.push_back(partial_per_example_gradient.index({offset}).data_ptr());
      offset += descriptors.at(i).grad_weight_per_example_size;
    }
    std::vector<void *> wgrads_ptrs2;
    offset = 0;
    for (size_t i = 0; i < descriptors.size(); ++i) {
      wgrads_ptrs2.push_back(partial_per_example_gradient2.index({offset}).data_ptr());
      offset += descriptors.at(i).grad_weight_per_example_size;
    }
    set_cutlass_device_ptrs(end_cudnn_layer, batch_count, actvs, ograds, wgrads_ptrs, wgrads_ptrs2);
    // checkCudaErrors(cudaDeviceSynchronize());
  }

  for (int example_idx = 0; example_idx < batch_count; ++example_idx) {
    // Compute wgrad of front layers using cuDNN
    compute_single_per_example_gradient_cudnn(descriptors, partial_per_example_gradient, actvs, ograds, end_cudnn_layer, example_idx);

    compute_single_per_example_gradient_cutlass(descriptors, end_cudnn_layer, batch_count, example_idx);

    checkCudaErrors(cudaDeviceSynchronize());

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
                                                                  int batch_count = 0,
                                                                  float max_norm = 1.0,
                                                                  bool quant = false,
                                                                  bool verbose = false,
                                                                  bool time_profile = false,
                                                                  bool memory_profile = false) {

  using namespace torch::indexing;

  auto start_time = std::chrono::high_resolution_clock::now();
  float backward_weight_ms = 0.0;
  float norm_ms = 0.0;
  float clip_reduce_ms = 0.0;

  auto partial_per_example_gradient = torch::zeros({(int)partial_per_example_gradient_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  auto partial_per_example_gradient2 = torch::zeros({(int)partial_per_example_gradient_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  
  // Set CUTLASS device pointers
  if (descriptors.size() - end_cudnn_layer > 0) {
    std::vector<void *> wgrads_ptrs;
    int64_t offset = 0;
    for (size_t i = 0; i < descriptors.size(); ++i) {
      wgrads_ptrs.push_back(partial_per_example_gradient.index({offset}).data_ptr());
      offset += descriptors.at(i).grad_weight_per_example_size;
    }
    std::vector<void *> wgrads_ptrs2;
    offset = 0;
    for (size_t i = 0; i < descriptors.size(); ++i) {
      wgrads_ptrs2.push_back(partial_per_example_gradient2.index({offset}).data_ptr());
      offset += descriptors.at(i).grad_weight_per_example_size;
    }
    set_cutlass_device_ptrs(end_cudnn_layer, batch_count, actvs, ograds, wgrads_ptrs, wgrads_ptrs2);
    // checkCudaErrors(cudaDeviceSynchronize());
  }

  auto partial_per_example_gradient_decoded = torch::zeros({(int)partial_per_example_gradient_size}, torch::TensorOptions().device(torch::kCUDA, 0));
  auto per_batch_gradient = torch::zeros({(int)non_reweight_per_example_gradient_size}, torch::TensorOptions().device(torch::kCUDA, 0));

  // Workspace to store scaling factors
  auto scaling_factors = torch::empty({batch_count}, torch::TensorOptions().device(torch::kCUDA, 0));

  // Workspace to store partial sums during norm computation
  auto partial_sums = torch::zeros({1 + (int)precomputed_per_example_grad_norms.size()}, torch::TensorOptions().device(torch::kCUDA, 0));

  LOG_STDERR("Compute per-example gradients and scaling factors", verbose);
  // Compute per-example gradients and scaling factors
  for (int example_idx = 0; example_idx < batch_count; ++example_idx) {
    // Compute wgrad of front layers using cuDNN
    compute_single_per_example_gradient_cudnn(descriptors, partial_per_example_gradient, actvs, ograds, end_cudnn_layer, example_idx);
    // torch::cuda::synchronize();
    // checkCudaErrors(cudaDeviceSynchronize());
    LOG_STDERR("Compute wgrad of back layers using CUTLASS", verbose);
    // Compute wgrad of back layers using CUTLASS
    compute_single_per_example_gradient_cutlass(descriptors, end_cudnn_layer, batch_count, example_idx);

    // checkCudaErrors(cudaDeviceSynchronize());
    // torch::cuda::synchronize();
    TIME_PROFILE(backward_weight_ms, time_profile);

    partial_per_example_gradient.index_put_({(int)partial_per_example_gradient_size}, precomputed_per_example_grad_norms.at(0).index({0}));
    // torch::cuda::synchronize();
    LOG_STDERR("Compute scaling factor", verbose);
    // Compute scaling factor
    if (quant) {
      partial_per_example_gradient_decoded = partial_per_example_gradient.toType(torch::kFloat);
    }
    compute_single_scaling_factor(scaling_factors, example_idx, partial_sums,
                                  partial_per_example_gradient, precomputed_per_example_grad_norms, max_norm);
    // torch::cuda::synchronize();
    LOG_STDERR("Clip and accumulate", verbose);
    // Clip and accumulate
    per_batch_gradient += partial_per_example_gradient.index({Slice(0, (int)non_reweight_per_example_gradient_size)}) * scaling_factors.index({example_idx});

    // torch::cuda::synchronize();
    TIME_PROFILE(clip_reduce_ms, time_profile);
  }

  // Scale and reduce pre-computed per-example grads
  LOG_STDERR("Scale and reduce pre-computed per-example grads", verbose);
  std::vector<torch::Tensor> per_batch_grads_from_precomputed;
  for (auto grad : precomputed_per_example_grads) {
    std::vector<int64_t> scaling_factors_shape;
    scaling_factors_shape.push_back(scaling_factors.size(0));
    for (size_t i = 0; i < grad.sizes().size() - 1; ++i){
      scaling_factors_shape.push_back(1);
    }
    per_batch_grads_from_precomputed.push_back(torch::sum(grad * scaling_factors.view(scaling_factors_shape), {0}));
  }

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
    // torch::mul_out(actvs.at(i), actvs.at(i), scaling_factors.view(scaling_factors_shape));
    torch::mul_out(ograds.at(i), ograds.at(i), scaling_factors.view(scaling_factors_shape));
  }
  
  LOG_STDERR("Scale output grads (for linear layer) for reweight", verbose);
  for (size_t i = 0; i < linear_actvs.size(); ++i) {
    std::vector<int64_t> scaling_factors_shape;
    scaling_factors_shape.push_back(scaling_factors.size(0));
    for (size_t j = 0; j < linear_actvs.at(i).sizes().size() - 1; ++j){
      scaling_factors_shape.push_back(1);
    }
    // torch::mul_out(linear_actvs.at(i), linear_actvs.at(i), scaling_factors.view(scaling_factors_shape));
    torch::mul_out(linear_ograds.at(i), linear_ograds.at(i), scaling_factors.view(scaling_factors_shape));
  }

  std::vector<torch::Tensor> per_batch_grads;
  // Split finished per-batch gradients and add to list
  LOG_STDERR("Split finished per-batch gradients and add to list", verbose);
  int64_t offset = 0;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    if (i < end_non_reweight_layer) {
      per_batch_grads.push_back(per_batch_gradient.index({Slice(offset, offset + descriptors.at(i).grad_weight_per_example_size)}));
      offset += descriptors.at(i).grad_weight_per_example_size;
    }
    else {
      per_batch_grads.push_back(torch::empty(descriptors.at(i).filter_shape, torch::TensorOptions().device(torch::kCUDA, 0)));
    }
  }
  assert(per_batch_grads.size() == descriptors.size());

  LOG_STDERR("Compute per-batch gradient for rewight layers", verbose);
  int alpha = 1;
  int beta = 0;
  for (size_t i = 0; i < descriptors.size(); ++i) {
    if (i < end_non_reweight_layer){
      continue;
    }
    // Compute per-batch gradient for rewight layers
    Conv2dDescriptor& descriptor = descriptors.at(i);

    checkCUDNN(cudnnConvolutionBackwardFilter(at::native::getCudnnHandle(),
                                              &alpha,
                                              descriptor.input_batch_desc,
                                              actvs.at(i).data_ptr(),
                                              descriptor.output_batch_desc,
                                              ograds.at(i).data_ptr(),
                                              descriptor.conv_desc,
                                              descriptor.bwd_filter_batch_algo_best,
                                              bwd_filter_batch_workspace,
                                              max_bwd_filter_batch_algo_best_workspace_size,
                                              &beta,
                                              descriptor.filter_desc,
                                              per_batch_grads.at(i).data_ptr()));
  }

  // checkCudaErrors(cudaDeviceSynchronize());

  std::vector<torch::Tensor> per_batch_linear_grads;
  // Split finished per-batch gradients and add to list
  LOG_STDERR("Split finished per-batch gradients and add to list", verbose);
  for (size_t i = 0; i < linear_actvs.size(); ++i) {
    per_batch_linear_grads.push_back(torch::empty({linear_ograds.at(i).sizes()[1], linear_actvs.at(i).sizes()[1]}, torch::TensorOptions().device(torch::kCUDA, 0)));
  }
  for (size_t i = 0; i < linear_actvs.size(); ++i) {
    torch::matmul_out(per_batch_linear_grads.at(i), linear_ograds.at(i).transpose(0, 1), linear_actvs.at(i));
  }

  TIME_PROFILE(clip_reduce_ms, time_profile);

  return ReturnType({per_batch_grads, per_batch_grads_from_precomputed, per_batch_linear_grads}, backward_weight_ms, norm_ms, clip_reduce_ms, 0);
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
                                          int batch_count,
                                          float max_norm,
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
    for (size_t end_cudnn_layer = 0; end_cudnn_layer < configs.size() + 1; ++end_cudnn_layer) {
      // Initialize cutlass_wgrad_grouped library
      cutlass_wgrad_grouped::initialize();

      // Set cudnn workspace and compute non-reweight per-example gradient size
      set_cudnn_workspace(end_cudnn_layer);

      // Set cutlass workspace
      set_cutlass_workspace(end_cudnn_layer, batch_count);

      // Set cutlass best operation
      set_cutlass_best_operation(end_cudnn_layer, batch_count, actvs, ograds);
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
      for (int i = 0; i < 10; ++i) {
        benchmark_cudnn_and_cutlass(actvs, ograds, end_cudnn_layer); 
      }

      checkCudaErrors(cudaDeviceSynchronize());
      auto startTime = std::chrono::high_resolution_clock::now();

      // Run clip_and_reduce_grads
      for (int i = 0; i < 10; ++i) {
        benchmark_cudnn_and_cutlass(actvs, ograds, end_cudnn_layer);
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
      if (increase_count == THRESHOLD_INCREASE_COUNT) {
        break;
      }
      prev_runtime_us = runtime_us;

      std::ostringstream stringStream;
      stringStream << "Current end cudnn layer = " << end_cudnn_layer << ", runtime = " << runtime_us.count() << " us";
      std::string copyOfStr = stringStream.str();
      LOG_STDERR(copyOfStr, true);
    }

    {
      std::ostringstream stringStream;
      stringStream << "Best end-cudnn-layer = " << best_end_cudnn_layer;
      std::string copyOfStr = stringStream.str();
      LOG_STDERR(copyOfStr, true);
    }

    // Initialize cutlass_wgrad_grouped library
    cutlass_wgrad_grouped::initialize();

    // Set cutlass workspace with best end-cudnn-layer
    set_cutlass_workspace(best_end_cudnn_layer, batch_count);

    // Set cutlass best operation with best end-cudnn-layer
    set_cutlass_best_operation(best_end_cudnn_layer, batch_count, actvs, ograds);

    ///////////////////////////////////////////////////////////////////////////////////
    
    // Profile to find best end-non-reweight-layer

    LOG_STDERR("Profile...", verbose);

    min_runtime_us = std::chrono::microseconds(1000000000);
    prev_runtime_us = std::chrono::microseconds(1000000000);
    increase_count = 0;
    
    // Profile to find best end-non-reweight-layer
    for (size_t end_non_reweight_layer = 0; end_non_reweight_layer < configs.size() + 1; ++end_non_reweight_layer) {

      // Compute non-reweight per-example gradient size
      non_reweight_per_example_gradient_size = 0;
      for (size_t i = 0; i < configs.size(); ++i) {
        if (i < end_non_reweight_layer) {
          non_reweight_per_example_gradient_size += (size_t)configs.at(i).K*configs.at(i).C*configs.at(i).R*configs.at(i).S;
        }
      }

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
                                            batch_count,
                                            max_norm,
                                            quant,
                                            false,
                                            false,
                                            false); 
      }

      checkCudaErrors(cudaDeviceSynchronize());
      auto startTime = std::chrono::high_resolution_clock::now();

      // Run clip_and_reduce_grads
      for (int i = 0; i < 10; ++i) {
        auto _ = clip_and_reduce_grads_conv(configs,
                                            actvs,
                                            ograds,
                                            precomputed_per_example_grads,
                                            precomputed_per_example_grad_norms,
                                            linear_actvs,
                                            linear_ograds,
                                            end_non_reweight_layer,
                                            best_end_cudnn_layer,
                                            batch_count,
                                            max_norm,
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
      if (increase_count == THRESHOLD_INCREASE_COUNT) {
        break;
      }
      prev_runtime_us = runtime_us;

      std::ostringstream stringStream;
      stringStream << "Current end non-reweight layer = " << end_non_reweight_layer << ", runtime = " << runtime_us.count() << " us";
      std::string copyOfStr = stringStream.str();
      LOG_STDERR(copyOfStr, true);
    }

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
  }

  _first_run = false;

  return clip_and_reduce_grads_conv(configs,
                                    actvs,
                                    ograds,
                                    precomputed_per_example_grads,
                                    precomputed_per_example_grad_norms,
                                    linear_actvs,
                                    linear_ograds,
                                    best_end_non_reweight_layer,
                                    best_end_cudnn_layer,
                                    batch_count,
                                    max_norm,
                                    quant,
                                    verbose,
                                    profile_time,
                                    profile_memory);
}