#include <torch/extension.h>
#include "ATen/cudnn/Handles.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <vector>
#include <chrono>
#include <algorithm>

#include "structure.h"
#include "error_helper.h"

#include "cutlass_wgrad_grouped.h"

#include "cutlass_simt_int8_batched_gemm.h"
#include "compute_scaling_factor_cuda.h"

#include "grad_example_module_linear.h"

#include "utils.h"
#include "quantize.h"
#include "add_noise.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global variables

#define _PTR_ALIGN_SIZE 256

namespace Linear{
bool _first_run = true;

std::vector<LinearDescriptor> descriptors;

int64_t partial_per_example_gradient_size = 0;
size_t non_reweight_per_example_gradient_size = 0;
size_t best_end_non_reweight_layer = 3;

int n_streams = 10;

// #define _N_CUDA_STREAMS 100
cudaStream_t cuda_streams[10];
cublasHandle_t cublas_handles[10];
cublasHandle_t cublas_device_handles[10];

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_descriptors_linear(std::vector<LinearConfig> &configs, bool quant=false, int end_non_reweight_layer=3) {
  // Create custom cuda streams
  for (int i=0; i < Linear::n_streams; ++i) {
    checkCudaErrors(cudaStreamCreate(&Linear::cuda_streams[i]));
    checkCUBLAS(cublasCreate(&Linear::cublas_handles[i]));
    checkCUBLAS(cublasSetStream(Linear::cublas_handles[i], Linear::cuda_streams[i]));
  }
  cublasSetPointerMode(Linear::cublas_handles[9], CUBLAS_POINTER_MODE_DEVICE);
  for (int i=0; i < Linear::n_streams; ++i) {
    checkCUBLAS(cublasCreate(&Linear::cublas_device_handles[i]));
    if (i > 0) {
      checkCUBLAS(cublasSetStream(Linear::cublas_device_handles[i], Linear::cuda_streams[i]));
    }
    cublasSetPointerMode(Linear::cublas_device_handles[i], CUBLAS_POINTER_MODE_DEVICE);
  }

  Linear::descriptors.clear();
  Linear::descriptors.resize(configs.size());

  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;

  size_t total_ws_size = 0;
  for (size_t i=0; i < configs.size(); ++i) {
    auto config = &configs.at(i);
    // Generate cublas descriptors
    Linear::descriptors.at(i).grad_weight_per_example_size_in_bytes += (size_t)sizeof(float)\
                                        *config->in_features\
                                        *config->out_features;
    Linear::descriptors.at(i).grad_weight_per_example_size += config->in_features\
                                                      *config->out_features;

    Linear::descriptors.at(i).config = {config->N,
                                config->seq_len,
                                config->in_features,
                                config->out_features,
                                config->num_layers};

    int64_t gradient_size = (int64_t)Linear::descriptors.at(i).grad_weight_per_example_size;
    for (int layer_idx=0; layer_idx < config->num_layers; ++layer_idx) {
      Linear::partial_per_example_gradient_size += gradient_size;
    }

    Linear::descriptors.at(i).weight_shape = {config->out_features, config->in_features};

    // To store ptrs for CUBLAS and CUTLASS
    Linear::descriptors.at(i).tensor_for_A_array = torch::empty({(int64_t)sizeof(void *)*config->num_layers*config->N}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
    Linear::descriptors.at(i).tensor_for_B_array = torch::empty({(int64_t)sizeof(void *)*config->num_layers*config->N}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
    Linear::descriptors.at(i).tensor_for_C_array = torch::empty({(int64_t)sizeof(void *)*config->num_layers}, torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kChar));
    Linear::descriptors.at(i).A_array = (void **)Linear::descriptors.at(i).tensor_for_A_array.data_ptr();
    Linear::descriptors.at(i).B_array = (void **)Linear::descriptors.at(i).tensor_for_B_array.data_ptr();
    Linear::descriptors.at(i).C_array = (void **)Linear::descriptors.at(i).tensor_for_C_array.data_ptr();

    Linear::descriptors.at(i).host_A_array.resize(config->N*config->num_layers);
    Linear::descriptors.at(i).host_B_array.resize(config->N*config->num_layers);
    Linear::descriptors.at(i).host_C_array.resize(config->num_layers);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename dType>
void set_cublas_device_ptr_array(std::vector<std::vector<torch::Tensor>>& actvs,
                                 std::vector<std::vector<torch::Tensor>>& ograds,
                                 std::vector<std::vector<void *>>& wgrad_ptrs) {
    
    for (size_t descriptor_idx=0; descriptor_idx < Linear::descriptors.size(); ++descriptor_idx) {
        LinearDescriptor& descriptor = Linear::descriptors.at(descriptor_idx);

        int batch_size = descriptor.config.N;
        int num_layers = descriptor.config.num_layers;

        std::vector<void *>& host_A_array = descriptor.host_A_array;
        std::vector<void *>& host_B_array = descriptor.host_B_array;
        std::vector<void *>& host_C_array = descriptor.host_C_array;

        for (size_t layer_idx=0; layer_idx < num_layers; ++layer_idx) {
          auto& current_actvs = actvs.at(descriptor_idx).at(layer_idx);
          auto& current_ograds = ograds.at(descriptor_idx).at(layer_idx);

          dType * A_ptr = (dType *)current_actvs.data_ptr();
          dType * B_ptr = (dType *)current_ograds.data_ptr();
          size_t A_example_offset = current_actvs.numel() / current_actvs.sizes()[0];
          size_t B_example_offset = current_ograds.numel() / current_ograds.sizes()[0];

          for (size_t example_idx=0; example_idx < batch_size; ++example_idx) {
              host_A_array.at(example_idx*num_layers + layer_idx) = A_ptr;
              host_B_array.at(example_idx*num_layers + layer_idx) = B_ptr;
              A_ptr += A_example_offset;
              B_ptr += B_example_offset;
          }
        }

        for (size_t layer_idx=0; layer_idx < num_layers; ++layer_idx) {
            host_C_array.at(layer_idx) = wgrad_ptrs.at(descriptor_idx).at(layer_idx);
        }

        checkCudaErrors(cudaMemcpyAsync((void *)descriptor.A_array, &host_A_array[0], sizeof(void*)*batch_size*num_layers, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync((void *)descriptor.B_array, &host_B_array[0], sizeof(void*)*batch_size*num_layers, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync((void *)descriptor.C_array, &host_C_array[0], sizeof(void*)*num_layers, cudaMemcpyHostToDevice));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_single_per_example_gradient_cublas(torch::Tensor &per_example_gradient,
                                                torch::Tensor &per_example_gradient_int32,
                                                std::vector<std::vector<torch::Tensor>>& actvs,
                                                std::vector<std::vector<torch::Tensor>>& ograds,
                                                int example_idx,
                                                bool quant=false) {
  assert(Linear::descriptors.size() == actvs.size());
  assert(Linear::descriptors.size() == ograds.size());

  using namespace torch::indexing;

  // cudnnHandle_t cudnn_handle;
  float alpha_float = 1.0;
  float beta_float = 0.0;

  int offset = 0;
  
  for (size_t i=0; i < Linear::descriptors.size(); ++i) {
    LinearDescriptor& descriptor = Linear::descriptors.at(i);

    int32_t alpha_int = 1;
    int32_t beta_int = 0;
    void * alpha = quant ? (void *)&alpha_int : (void *)&alpha_float;
    void * beta = quant ? (void *)&beta_int : (void *)&beta_float;

    int m = descriptor.config.in_features;
    int n = descriptor.config.out_features;
    int k = descriptor.config.seq_len;
    int batch_size = descriptor.config.N;
    int num_layers = descriptor.config.num_layers;

    if (quant) {
      // Column major X Column major -> Column major
      checkCudaErrors(cutlass_simt_igemm_int8_batched_gemm(m, n, k,
                                                           alpha_float,
                                                           (int8_t **)descriptor.A_array + example_idx*num_layers,
                                                           k,
                                                           (int8_t **)descriptor.B_array + example_idx*num_layers,
                                                           k,
                                                           (float **)descriptor.C_array, m,
                                                           beta_float,
                                                           num_layers,
                                                           Linear::cuda_streams[i % Linear::n_streams]
                                                           ));
    }
    else {
      checkCUBLAS(cublasGemmBatchedEx(Linear::cublas_handles[i % Linear::n_streams],
                                      CUBLAS_OP_N, CUBLAS_OP_T,
                                      m, n, k,
                                      alpha,
                                      descriptor.A_array + example_idx*num_layers,
                                      CUDA_R_32F,
                                      m,
                                      descriptor.B_array + example_idx*num_layers,
                                      CUDA_R_32F,
                                      n,
                                      beta,
                                      descriptor.C_array,
                                      CUDA_R_32F,
                                      m,
                                      num_layers,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_single_per_example_gradient_embedding(torch::Tensor embedding_gradients, 
                                                   std::vector<torch::Tensor>& embedding_actvs,
                                                   std::vector<torch::Tensor>& embedding_ograds,
                                                   std::vector<int>& embedding_vocab_sizes,
                                                   int example_idx) {

    using namespace torch::indexing;


    int64_t offset = Linear::partial_per_example_gradient_size;
    for (size_t i=0; i < embedding_actvs.size(); ++i) {
        int embedding_gradient_size = embedding_ograds.at(i).sizes()[2] * embedding_vocab_sizes.at(i);
        auto embedding_gradient = embedding_gradients.index({Slice(offset, offset + embedding_gradient_size)}).view({embedding_vocab_sizes.at(i), embedding_ograds.at(i).sizes()[2]});
        // printf("embedding_gradient %p\n", embedding_gradient.data_ptr());

        auto actv = embedding_actvs.at(i).index({example_idx});
        auto ograd = embedding_ograds.at(i).index({example_idx});
        auto index = actv.unsqueeze(-1).expand({actv.sizes()[0], ograd.sizes()[1]}).reshape({-1, ograd.sizes()[1]});

        // embedding_gradient.zero_();
        // printf("embedding_gradient2 %p\n", embedding_gradient.data_ptr());
        checkCudaErrors(cudaMemsetAsync(embedding_gradient.data_ptr(), 0, embedding_gradient_size * sizeof(float)));
        // embedding_gradient.index_put_({Slice()}, torch::zeros({embedding_vocab_sizes.at(i), ograd.sizes()[1]}, torch::TensorOptions().device(torch::kCUDA, 0)));

        embedding_gradient.scatter_add_(0, index, ograd.reshape({actv.sizes()[0], ograd.sizes()[1]}));
    
        offset += embedding_gradient_size;

        // std::cout << "Embedding norm, " << torch::frobenius_norm(embedding_gradient.flatten(), {0}, false) << std::endl;
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_per_batch_gradient_embedding(std::vector<torch::Tensor>& per_batch_gradients, 
                                          std::vector<torch::Tensor>& embedding_actvs,
                                          std::vector<torch::Tensor>& embedding_ograds,
                                          std::vector<int>& embedding_vocab_sizes) {

    using namespace torch::indexing;

    int offset = 0;
    for (size_t i=0; i < embedding_actvs.size(); ++i) {
        auto embedding_gradient = per_batch_gradients.at(i);

        auto actv = embedding_actvs.at(i);
        auto ograd = embedding_ograds.at(i);
        auto index = actv.unsqueeze(-1).expand({actv.sizes()[0], actv.sizes()[1], ograd.sizes()[2]}).reshape({-1, ograd.sizes()[2]});

        embedding_gradient.zero_();

        embedding_gradient.scatter_add_(0, index, ograd.reshape({ograd.sizes()[0]*ograd.sizes()[1], ograd.sizes()[2]}));
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// torch::Tensor prev_partial_per_example_gradient = torch::empty({0});

void compute_single_scaling_factor(float * scaling_factor_out,
                                   const torch::Tensor partial_per_example_gradient,
                                   float max_norm,
                                   int scale_loss) {
  
  using namespace torch::indexing;
  // std::cout << "pre_computed_grad_norms, " << partial_per_example_gradient.index({-1}) << std::endl;
  // if (prev_partial_per_example_gradient.sizes()[0] != 0) {
  //   // std::cout << "partial_per_example_gradient_size, " << Linear::partial_per_example_gradient_size << std::endl;
  //   // std::cout << "Non equals, " << prev_partial_per_example_gradient.ne(partial_per_example_gradient).nonzero().sizes()[0] << std::endl;
  //   if (prev_partial_per_example_gradient.ne(partial_per_example_gradient).nonzero().sizes()[0] > 0) {
  //     std::cout << "Non equals, " << prev_partial_per_example_gradient.ne(partial_per_example_gradient).nonzero()[0].item() << std::endl;
  //   }
  // }
  // std::cout << "partial_per_example_gradient, " << partial_per_example_gradient.index({Slice(91034112, 91034112+10)}) << std::endl;
  // prev_partial_per_example_gradient = partial_per_example_gradient;
  auto norm = torch::frobenius_norm(partial_per_example_gradient, {0}, false);
  // std::cout << "Norms, " << norm << std::endl;
  compute_scaling_factor_cuda(scaling_factor_out, (float *)norm.data_ptr(), max_norm, 1, scale_loss);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ReturnType clip_and_reduce_grads_linear(std::vector<LinearConfig> &configs,
                                    std::vector<std::vector<torch::Tensor>>& actvs,
                                    std::vector<std::vector<torch::Tensor>>& ograds,
                                    std::vector<torch::Tensor>& precomputed_per_example_grads,
                                    std::vector<torch::Tensor>& precomputed_per_example_grad_norms,
                                    std::vector<torch::Tensor>& linear_last_actvs,
                                    std::vector<torch::Tensor>& linear_last_ograds,
                                    std::vector<torch::Tensor>& embedding_actvs,
                                    std::vector<torch::Tensor>& embedding_ograds,
                                    std::vector<int>& embedding_vocab_sizes,
                                    size_t end_non_reweight_layer,
                                    bool loss_reduction_mean = false,
                                    int batch_count = 0,
                                    float max_norm = 1.0,
                                    float noise_multiplier = 1.0,
                                    bool quant = false,
                                    bool verbose = false,
                                    bool time_profile = false,
                                    bool memory_profile = false) {

  using namespace torch::indexing;

  // Variables for profiling timing
  auto start_time = std::chrono::high_resolution_clock::now();
  float backward_weight_ms = 0.0;
  float clip_reduce_ms = 0.0;
  float norm_ms = 0.0;
  float add_noise_ms = 0.0;

  for (auto& ograds_flatten : ograds) {
    for (auto& ograd : ograds_flatten) {
      if (!ograd.is_contiguous()) {
        ograd = ograd.contiguous();
      }
    }
  }
  
  // Create tensor for embedding gradients
  int embedding_gradient_size = 0;
  std::vector<torch::Tensor> embedding_per_example_gradients;
  for (size_t i=0; i<embedding_actvs.size(); ++i) {
    embedding_gradient_size += embedding_vocab_sizes.at(i) * embedding_ograds.at(i).sizes()[2];
    // printf("embedding_gradient_size = %d\n", embedding_gradient_size);
  }
  // printf("1 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  auto partial_per_example_gradient = torch::empty({(int64_t)Linear::partial_per_example_gradient_size + embedding_gradient_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  // printf("2 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  // Put precomputed_norm at last to compute norm of total gradient
  // To compute clip and reduce efficiently, gather all precomputed grads into a single tensor
  int64_t gathered_per_example_grads_size = 0;
  for (size_t i = 0; i < precomputed_per_example_grads.size(); ++i) {
    gathered_per_example_grads_size += precomputed_per_example_grads.at(i).numel() / batch_count;
  }
  auto gathered_per_example_grads = torch::empty({batch_count, gathered_per_example_grads_size + 1}, torch::TensorOptions().device(torch::kCUDA, 0));
  {
    int64_t offset = 0;
    for (size_t i = 0; i < precomputed_per_example_grads.size(); ++i) {
      gathered_per_example_grads.index_put_({Slice(), Slice(offset, offset + precomputed_per_example_grads.at(i).numel() / batch_count)},
                                            precomputed_per_example_grads.at(i).view({batch_count, -1}));
      offset += precomputed_per_example_grads.at(i).numel() / batch_count;
    }
  }
  // std::cout << gathered_per_example_grads.index({0, Slice(0, 10)}) << std::endl;
  gathered_per_example_grads.index_put_({Slice(), gathered_per_example_grads_size}, precomputed_per_example_grad_norms.at(0));
  auto precomputed_per_example_norms = torch::frobenius_norm(gathered_per_example_grads, {1});
  gathered_per_example_grads = gathered_per_example_grads.index({Slice(0, batch_count), Slice(0, gathered_per_example_grads_size)});
  // printf("3 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  // Set CUTLASS device pointers
  std::vector<std::vector<void *>> wgrad_ptrs;
  wgrad_ptrs.resize(Linear::descriptors.size());
  int64_t offset = 0;
  for (size_t i = 0; i < Linear::descriptors.size(); ++i) {
    for (size_t layer_idx=0; layer_idx < Linear::descriptors.at(i).config.num_layers; ++layer_idx) {
      wgrad_ptrs.at(i).push_back(partial_per_example_gradient.index({offset}).data_ptr());
      
      offset += ((Linear::descriptors.at(i).grad_weight_per_example_size));
    }
  }
  if (quant) {
    set_cublas_device_ptr_array<int8_t>(actvs, ograds, wgrad_ptrs);
  }
  else{
    set_cublas_device_ptr_array<float>(actvs, ograds, wgrad_ptrs);
  }
  // checkCudaErrors(cudaDeviceSynchronize());
  // printf("4 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  // c10::cuda::CUDACachingAllocator::emptyCache();
  // Create tensor to accumulate gradients
  auto per_batch_gradient = torch::zeros({(int64_t)Linear::partial_per_example_gradient_size}, torch::TensorOptions().device(torch::kCUDA, 0));
  auto partial_per_batch_gradient = per_batch_gradient.index({Slice(0, (int64_t)Linear::non_reweight_per_example_gradient_size)});
  // c10::cuda::CUDACachingAllocator::emptyCache();
  // printf("4.1 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  // Workspace to store scaling factors
  auto scaling_factors = torch::empty({batch_count}, torch::TensorOptions().device(torch::kCUDA, 0));

  TIME_PROFILE(clip_reduce_ms, time_profile);

  LOG_STDERR("Compute per-example gradients and scaling factors", verbose);

  // Create cuda events
  std::vector<cudaEvent_t> events(Linear::n_streams);
  for (size_t i=0; i < Linear::n_streams; ++i) {
    checkCudaErrors(cudaEventCreate(&events[i]));
  }
  // Compute per-example gradients and scaling factors
  for (int example_idx = 0; example_idx < batch_count; ++example_idx) {
    // partial_per_example_gradient.index_put_({(int64_t)Linear::partial_per_example_gradient_size + embedding_gradient_size}, precomputed_per_example_norms.index({example_idx}));
    cudaMemcpyAsync((void *)((float *)partial_per_example_gradient.data_ptr() + Linear::partial_per_example_gradient_size + embedding_gradient_size),
                    (void *)((float *)precomputed_per_example_norms.data_ptr() + example_idx),
                    4, cudaMemcpyDeviceToDevice);
    // Compute wgrad of front layers using CUBLAS
    compute_single_per_example_gradient_embedding(partial_per_example_gradient, embedding_actvs, embedding_ograds, embedding_vocab_sizes, example_idx);

    checkCudaErrors(cudaEventRecord(events[0], NULL));
    for (size_t i=0; i < configs.size(); ++i) {
      checkCudaErrors(cudaStreamWaitEvent(Linear::cuda_streams[i], events[0], 0));
    }

    compute_single_per_example_gradient_cublas(partial_per_example_gradient, partial_per_example_gradient, actvs, ograds, example_idx, quant);

    // Wait all streams to finish
    for (size_t i=0; i < configs.size(); ++i) {
      checkCudaErrors(cudaEventRecord(events[i], Linear::cuda_streams[i]));
    }
    for (size_t i=0; i < configs.size(); ++i) {
      checkCudaErrors(cudaStreamWaitEvent(NULL, events[i], 0));
    }

    TIME_PROFILE(backward_weight_ms, time_profile);

    LOG_STDERR("Compute scaling factor", verbose);
    // Compute scaling factor
    compute_single_scaling_factor((float *)scaling_factors.index({example_idx}).data_ptr(), 
                                  partial_per_example_gradient, max_norm, loss_reduction_mean ? batch_count : 1);

    LOG_STDERR("Clip and accumulate", verbose);
    // Clip and accumulate
    if (Linear::non_reweight_per_example_gradient_size > 0) {
      // printf("non_reweight_per_example_gradient_size = %d\n", Linear::non_reweight_per_example_gradient_size);
      partial_per_batch_gradient.add_(partial_per_example_gradient.index({Slice(0, (int64_t)Linear::non_reweight_per_example_gradient_size)}), scaling_factors.index({example_idx}).item());

      // checkCUBLAS(cublasSaxpy(Linear::cublas_device_handles[0],
      //                         (int)Linear::non_reweight_per_example_gradient_size,
      //                         (float *)scaling_factors.index({example_idx}).data_ptr(),
      //                         (float *)partial_per_example_gradient.data_ptr(),
      //                         1,
      //                         (float *)per_batch_gradient.data_ptr(),
      //                         1));
    }

    TIME_PROFILE(clip_reduce_ms, time_profile);
  }

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
  // per_batch_gradient_from_precomputed.add_(torch::normal(0.0, max_norm*noise_multiplier, per_batch_gradient_from_precomputed.sizes(), c10::nullopt, torch::TensorOptions().device(torch::kCUDA, 0)));
  add_noise(per_batch_gradient_from_precomputed, max_norm*noise_multiplier);
  TIME_PROFILE(add_noise_ms, time_profile);
  // printf("5 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
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

  std::vector<std::vector<torch::Tensor>> per_batch_grads;
  per_batch_grads.resize(Linear::descriptors.size());
  // Split finished per-batch gradients and add to list
  LOG_STDERR("Split finished per-batch gradients and add to list", verbose);
  offset = 0;
  for (size_t i = 0; i < Linear::descriptors.size(); ++i) {
    LinearDescriptor& desc = Linear::descriptors.at(i);

    for (size_t layer_idx = 0; layer_idx < desc.config.num_layers; ++layer_idx) {
      per_batch_grads.at(i).push_back(per_batch_gradient.index({Slice(offset, offset + desc.grad_weight_per_example_size)}).view(desc.weight_shape));
      offset += desc.grad_weight_per_example_size;
    }
  }
  assert(per_batch_grads.size() == Linear::descriptors.size());
//  printf("6 Peak memory usage %ld\n", c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes.at(0).peak);
  // Scale output grads
  LOG_STDERR("Scale output grads for reweight", verbose);
  for (size_t i = 0; i < Linear::descriptors.size(); ++i) {
    if (i < end_non_reweight_layer){
      continue;
    }
    std::vector<int64_t> scaling_factors_shape;
    scaling_factors_shape.push_back(scaling_factors.size(0));
    for (size_t j = 0; j < actvs.at(i).at(0).sizes().size() - 1; ++j){
      scaling_factors_shape.push_back(1);
    }
    for (int layer_idx = 0; layer_idx < ograds.at(i).size(); ++layer_idx) {
      auto& partial_ograds = ograds.at(i).at(layer_idx);

      int alpha = 1;
      int beta = 0;
      // if (quant) {
      //   partial_ograds = partial_ograds.to(torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32));
      // }
      
      at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(Linear::cuda_streams[layer_idx%Linear::n_streams], 0));

      torch::Tensor temp_ograds = torch::empty({0});
      torch::Tensor temp_actvs = torch::empty({0});
      if (quant) {
        temp_ograds = partial_ograds * scaling_factors.view(scaling_factors_shape);
        temp_actvs = actvs.at(i).at(layer_idx).to(torch::TensorOptions().dtype(torch::kFloat32));
      }
      else {
        temp_ograds = partial_ograds.mul(scaling_factors.view(scaling_factors_shape));
      }

      LOG_STDERR("Compute per-batch gradient for rewight layers", verbose);
      
      if (quant) {
        torch::matmul_out(per_batch_grads.at(i).at(layer_idx),
                          temp_ograds.reshape({temp_ograds.sizes()[0]*temp_ograds.sizes()[1], temp_ograds.sizes()[2]}).transpose(0, 1),
                          temp_actvs.reshape({temp_actvs.sizes()[0]*temp_actvs.sizes()[1], temp_actvs.sizes()[2]}));
      }
      else{
        torch::matmul_out(per_batch_grads.at(i).at(layer_idx),
                          temp_ograds.reshape({temp_ograds.sizes()[0]*temp_ograds.sizes()[1], temp_ograds.sizes()[2]}).transpose(0, 1),
                          actvs.at(i).at(layer_idx).reshape({actvs.at(i).at(layer_idx).sizes()[0]*actvs.at(i).at(layer_idx).sizes()[1], actvs.at(i).at(layer_idx).sizes()[2]}));
      }
    }
  }
  // Wait all streams to finish
  for (size_t i=0; i < Linear::n_streams; ++i) {
    checkCudaErrors(cudaEventRecord(events[i], Linear::cuda_streams[i]));
  }
  for (size_t i=0; i < Linear::n_streams; ++i) {
    checkCudaErrors(cudaStreamWaitEvent(NULL, events[i], 0));
  }
  at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());
  TIME_PROFILE(clip_reduce_ms, time_profile);
  // per_batch_gradient.add_(torch::normal(0.0, max_norm*noise_multiplier, per_batch_gradient.sizes(), c10::nullopt, torch::TensorOptions().device(torch::kCUDA, 0)));
  add_noise(per_batch_gradient, max_norm*noise_multiplier);
  TIME_PROFILE(add_noise_ms, time_profile);

  std::vector<torch::Tensor> per_batch_linear_last_grads;
  LOG_STDERR("Scale output grads (for linear layer) for reweight", verbose);
  for (size_t i = 0; i < linear_last_ograds.size(); ++i) {
    std::vector<int64_t> scaling_factors_shape;
    scaling_factors_shape.push_back(scaling_factors.size(0));
    for (size_t j = 0; j < linear_last_ograds.at(i).sizes().size() - 1; ++j) {
      scaling_factors_shape.push_back(1);
    }

    torch::Tensor temp_ograds = torch::empty({0});
    torch::Tensor temp_actvs = torch::empty({0});
    if (quant) {
      temp_ograds = linear_last_ograds.at(i) * scaling_factors.view(scaling_factors_shape);
      temp_actvs = linear_last_actvs.at(i).to(torch::TensorOptions().dtype(torch::kFloat32));
    }
    else {
      temp_ograds = linear_last_ograds.at(i).mul(scaling_factors.view(scaling_factors_shape));
    }

    LOG_STDERR("Compute per-batch gradient for linear last layers", verbose);
    per_batch_linear_last_grads.push_back(torch::empty({linear_last_ograds.at(i).sizes()[1], linear_last_actvs.at(i).sizes()[1]}, torch::TensorOptions().device(torch::kCUDA, 0)));
    if (quant) {
      torch::matmul_out(per_batch_linear_last_grads.at(i), temp_ograds.transpose(0, 1), temp_actvs);
    }
    else {
      torch::matmul_out(per_batch_linear_last_grads.at(i), temp_ograds.transpose(0, 1), linear_last_actvs.at(i));
    }
    TIME_PROFILE(clip_reduce_ms, time_profile);
    // per_batch_linear_last_grads.at(i).add_(torch::normal(0.0, max_norm*noise_multiplier, per_batch_linear_last_grads.at(i).sizes(), c10::nullopt, torch::TensorOptions().device(torch::kCUDA, 0)));
    add_noise(per_batch_linear_last_grads.at(i), max_norm*noise_multiplier);
    TIME_PROFILE(add_noise_ms, time_profile);
  }

  LOG_STDERR("Scale output grads (for embedding layer) for reweight", verbose);
  std::vector<torch::Tensor> scaled_embedding_ograds;
  for (size_t i = 0; i < embedding_ograds.size(); ++i) {
    std::vector<int64_t> scaling_factors_shape;
    scaling_factors_shape.push_back(scaling_factors.size(0));
    for (size_t j = 0; j < embedding_ograds.at(i).sizes().size() - 1; ++j) {
      scaling_factors_shape.push_back(1);
    }
    scaled_embedding_ograds.push_back(embedding_ograds.at(i).mul(scaling_factors.view(scaling_factors_shape)));
  }

  at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());

  // Flat per_batch_grads to std::vector<torch::Tensor>
  std::vector<torch::Tensor> flat_per_batch_grads;
  for (size_t i = 0; i < Linear::descriptors.size(); ++i) {
    for (size_t layer_idx = 0; layer_idx < Linear::descriptors.at(i).config.num_layers; ++layer_idx) {
      flat_per_batch_grads.push_back(per_batch_grads.at(i).at(layer_idx));
    }
  }
  // checkCudaErrors(cudaDeviceSynchronize());

  std::vector<torch::Tensor> per_batch_embedding_grads;
  // 
  LOG_STDERR("Scale output grads (for embedding layer) for reweight", verbose);
  LOG_STDERR("Compute per-batch gradient for embedding layers", verbose);
  for (size_t i = 0; i < embedding_actvs.size(); ++i) {
    per_batch_embedding_grads.push_back(torch::empty({embedding_vocab_sizes.at(i), embedding_ograds.at(i).sizes()[2]}, torch::TensorOptions().device(torch::kCUDA, 0)));
  }
  compute_per_batch_gradient_embedding(per_batch_embedding_grads, 
                                       embedding_actvs,
                                       scaled_embedding_ograds,
                                       embedding_vocab_sizes);
  TIME_PROFILE(clip_reduce_ms, time_profile);
  for (size_t i = 0; i < per_batch_embedding_grads.size(); ++i) {
    // per_batch_embedding_grads.at(i).add_(torch::normal(0.0, max_norm*noise_multiplier, per_batch_embedding_grads.at(i).sizes(), c10::nullopt, torch::TensorOptions().device(torch::kCUDA, 0)));
    add_noise(per_batch_embedding_grads.at(i), max_norm*noise_multiplier);
    
  }
  TIME_PROFILE(add_noise_ms, time_profile);

  TIME_PROFILE(clip_reduce_ms, time_profile);

  // printf("Finish C++, %lld\n", std::chrono::high_resolution_clock::now().time_since_epoch().count());

  return ReturnType({flat_per_batch_grads, per_batch_grads_from_precomputed, per_batch_linear_last_grads, per_batch_embedding_grads}, backward_weight_ms, norm_ms, clip_reduce_ms, add_noise_ms, 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ReturnType get_clip_and_reduced_grads_linear(std::vector<LinearConfig> &configs,
                                            std::vector<std::vector<torch::Tensor>>& actvs,
                                            std::vector<std::vector<torch::Tensor>>& ograds,
                                            std::vector<torch::Tensor>& precomputed_per_example_grads,
                                            std::vector<torch::Tensor>& precomputed_per_example_grad_norms,
                                            std::vector<torch::Tensor>& linear_last_actvs,
                                            std::vector<torch::Tensor>& linear_last_ograds,
                                            std::vector<torch::Tensor>& embedding_actvs,
                                            std::vector<torch::Tensor>& embedding_ograds,
                                            std::vector<int>& embedding_vocab_sizes,
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
  assert(linear_last_ograds.size() == linear_last_actvs.size());
  assert(embedding_ograds.size() == embedding_actvs.size());

  using namespace torch::indexing;

  if (Linear::_first_run) {

    LOG_STDERR("Set descriptor linear (first run)", verbose);
    set_descriptors_linear(configs, quant);
    
    ///////////////////////////////////////////////////////////////////////////////////

    // Profile to find best end-non-reweight-layer

    LOG_STDERR("Profile...", verbose);

    auto min_runtime_us = std::chrono::microseconds(1000000000);
    auto prev_runtime_us = std::chrono::microseconds(1000000000);
    auto increase_count = 0;
    
    // Profile to find best end-non-reweight-layer
    for (size_t end_non_reweight_layer = 0; end_non_reweight_layer < configs.size() + 1; ++end_non_reweight_layer) {

      // Compute non-reweight per-example gradient size
      Linear::non_reweight_per_example_gradient_size = 0;
      for (size_t i = 0; i < configs.size(); ++i) {
        if (i < end_non_reweight_layer) {
          for (size_t layer_idx = 0; layer_idx < configs.at(i).num_layers; ++layer_idx) {
            Linear::non_reweight_per_example_gradient_size += (size_t)configs.at(i).in_features*configs.at(i).out_features;
          }
        }
      }

      // Warm up
      for (int i = 0; i < 1; ++i) {
        auto _ = clip_and_reduce_grads_linear(configs,
                                              actvs,
                                              ograds,
                                              precomputed_per_example_grads,
                                              precomputed_per_example_grad_norms,
                                              linear_last_actvs,
                                              linear_last_ograds,
                                              embedding_actvs,
                                              embedding_ograds,
                                              embedding_vocab_sizes,
                                              end_non_reweight_layer,
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
      for (int i = 0; i < 10; ++i) {
        auto _ = clip_and_reduce_grads_linear(configs,
                                              actvs,
                                              ograds,
                                              precomputed_per_example_grads,
                                              precomputed_per_example_grad_norms,
                                              linear_last_actvs,
                                              linear_last_ograds,
                                              embedding_actvs,
                                              embedding_ograds,
                                              embedding_vocab_sizes,
                                              end_non_reweight_layer,
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
        Linear::best_end_non_reweight_layer = end_non_reweight_layer;
        min_runtime_us = runtime_us;
      }
      if (runtime_us <= prev_runtime_us) {
        increase_count = std::max(0, increase_count - 1);
      }
      else {
        increase_count += 1;
      }
      if (increase_count == 4) {
        break;
      }
      prev_runtime_us = runtime_us;

      std::ostringstream stringStream;
      stringStream << "Current end non-reweight layer = " << end_non_reweight_layer << ", runtime = " << runtime_us.count() << " us";
      std::string copyOfStr = stringStream.str();
      LOG_STDERR(copyOfStr, true);
    }

    std::ostringstream stringStream;
    stringStream << "Best end-non-reweight-layer = " << Linear::best_end_non_reweight_layer;
    std::string copyOfStr = stringStream.str();
    LOG_STDERR(copyOfStr, true);
    // Linear::best_end_non_reweight_layer = 4;// FIXME

    // Compute non-reweight per-example gradient size for best-non-reweight-layers
    Linear::non_reweight_per_example_gradient_size = 0;
    for (size_t i = 0; i < configs.size(); ++i) {
      if (i < Linear::best_end_non_reweight_layer) {
        for (size_t layer_idx = 0; layer_idx < configs.at(i).num_layers; ++layer_idx) {
          Linear::non_reweight_per_example_gradient_size += (size_t)configs.at(i).in_features*configs.at(i).out_features;
        }
      }
    }

  }

  Linear::_first_run = false;

  return clip_and_reduce_grads_linear(configs,
                                      actvs,
                                      ograds,
                                      precomputed_per_example_grads,
                                      precomputed_per_example_grad_norms,
                                      linear_last_actvs,
                                      linear_last_ograds,
                                      embedding_actvs,
                                      embedding_ograds,
                                      embedding_vocab_sizes,
                                      Linear::best_end_non_reweight_layer,
                                      loss_reduction_mean,
                                      batch_count,
                                      max_norm,
                                      noise_multiplier,
                                      quant,
                                      verbose,
                                      profile_time,
                                      profile_memory);
}