#pragma once
#include <torch/extension.h>
#include <cudnn.h>
#include "cutlass_wgrad_grouped.h"

using CutlassConv2dConfig = cutlass_wgrad_grouped::Conv2dConfig;

struct ReturnType {
  ReturnType() {};
  ReturnType(std::vector<std::vector<torch::Tensor>> per_batch_grads,
             float backward_weight_ms,
             float norm_ms,
             float clip_reduce_ms,
             float add_noise_ms,
             size_t peak_memory_usage):
             per_batch_grads(per_batch_grads),
             backward_weight_ms(backward_weight_ms),
             norm_ms(norm_ms),
             add_noise_ms(add_noise_ms),
             clip_reduce_ms(clip_reduce_ms),
             peak_memory_usage(peak_memory_usage) {};

  std::vector<std::vector<torch::Tensor>> get_per_batch_grads() {
    return per_batch_grads;
  }

  float get_backward_weight_ms() {
    return backward_weight_ms;
  }
  float get_norm_ms() {
    return norm_ms;
  }
  float get_clip_reduce_ms() {
    return clip_reduce_ms;
  } 
  float get_add_noise_ms() {
    return add_noise_ms;
  } 

  std::vector<std::vector<torch::Tensor>> per_batch_grads;
  
  float backward_weight_ms;
  float norm_ms;
  float clip_reduce_ms;
  float add_noise_ms;

  size_t peak_memory_usage;
};

struct LinearReturnType {
  LinearReturnType() {};
  LinearReturnType(std::vector<std::vector<torch::Tensor>> per_batch_grads,
             float backward_weight_ms,
             float norm_ms,
             float clip_reduce_ms,
             size_t peak_memory_usage):
             per_batch_grads(per_batch_grads),
             backward_weight_ms(backward_weight_ms),
             norm_ms(norm_ms),
             clip_reduce_ms(clip_reduce_ms),
             peak_memory_usage(peak_memory_usage) {};

  std::vector<std::vector<torch::Tensor>> get_per_batch_grads() {
    return per_batch_grads;
  }

  float get_backward_weight_ms() {
    return backward_weight_ms;
  }
  float get_norm_ms() {
    return norm_ms;
  }
  float get_clip_reduce_ms() {
    return clip_reduce_ms;
  } 

  std::vector<std::vector<torch::Tensor>> per_batch_grads;
  
  float backward_weight_ms;
  float norm_ms;
  float clip_reduce_ms;

  size_t peak_memory_usage;
};

struct Conv2dConfig {
  Conv2dConfig() {};
  Conv2dConfig(int N, int H, int W, int C,
               int K, int R, int S,
               int P, int Q,
               int pad_h, int pad_w,
               int stride_h, int stride_w,
               int dilation_h=1, int dilation_w=1,
               int split_k_slices=1):
    N(N), H(H), W(W), C(C),
    K(K), R(R), S(S),
    P(P), Q(Q),
    pad_h(pad_h), pad_w(pad_w),
    stride_h(stride_h), stride_w(stride_w),
    dilation_h(dilation_h), dilation_w(dilation_w),
    split_k_slices(split_k_slices) { };

  int N;
  int H;
  int W;
  int C;
  int K;
  int R;
  int S;
  int P;
  int Q;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int split_k_slices;
};

struct Conv2dDescriptor {
    Conv2dConfig config;
    // TensorCustom** input_actv;
    // TensorCustom** grad_output;
    // TensorCustom** grad_weight_per_example_ptr;
    size_t grad_weight_per_example_size_in_bytes = 0;
    size_t grad_weight_per_example_size;
    size_t workspace_size_in_bytes = 0;
    std::vector<int64_t> filter_shape;

    // Conv2D
    //// cuDNN
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_best;
    size_t bwd_filter_algo_best_workspace_size;
    void * workspace_ptr;
    torch::Tensor tensor_for_workspace_ptr;

    //// For per-batch reweight
    cudnnTensorDescriptor_t input_batch_desc;
    cudnnTensorDescriptor_t output_batch_desc;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_batch_algo_best;
    size_t bwd_filter_batch_algo_best_workspace_size;
    void * batch_workspace_ptr;
    torch::Tensor tensor_for_batch_workspace_ptr;

    //// cutlass
    CutlassConv2dConfig cutlass_config;
};


struct LinearConfig {
  LinearConfig() {};
  LinearConfig(int N,
               int seq_len, 
               int in_features,
               int out_features,
               int num_layers):
    N(N), seq_len(seq_len),
    in_features(in_features),
    out_features(out_features),
    num_layers(num_layers) { };

  int N;
  int seq_len;
  int in_features;
  int out_features;
  int num_layers;
};

struct LinearDescriptor {
    LinearConfig config;
    // TensorCustom** input_actv;
    // TensorCustom** grad_output;
    // TensorCustom** grad_weight_per_example_ptr;
    size_t grad_weight_per_example_size_in_bytes = 0;
    size_t grad_weight_per_example_size;
    size_t workspace_size_in_bytes = 0;
    std::vector<int64_t> weight_shape;

    // Linear
    //// cublas
    void ** A_array;
    void ** B_array;
    void ** C_array;
    torch::Tensor tensor_for_A_array;
    torch::Tensor tensor_for_B_array;
    torch::Tensor tensor_for_C_array;    

    std::vector<void *> host_A_array;
    std::vector<void *> host_B_array;
    std::vector<void *> host_C_array;

};