#include <torch/extension.h>
#include "ATen/cudnn/Handles.h"
#include <c10/cuda/CUDAStream.h>

#include <vector>
#include <chrono>
#include <algorithm>

#include <cublas_v2.h>

#include "error_helper.h"

std::vector<torch::Tensor> lstm_cuda_forward(
    torch::Tensor input, // [N, L, I]
    torch::Tensor weight_ih, // [4*H, I]
    torch::Tensor weight_hh, // [4*H, H]
    torch::Tensor bias, // [4*H]
    torch::Tensor weight_ih_reverse, // [4*H, I]
    torch::Tensor weight_hh_reverse, // [4*H, H]
    torch::Tensor bias_reverse, // [4*H]
    torch::Tensor old_h, // [D, N, H]
    torch::Tensor old_c, // [D, N, H]
    bool bidirectional = false
  );

std::vector<torch::Tensor> lstm_forward(
    torch::Tensor input, // [N, L, I]
    torch::Tensor weight_ih, // [4*H, I]
    torch::Tensor weight_hh, // [4*H, H]
    torch::Tensor bias, // [4*H]
    torch::Tensor weight_ih_reverse, // [4*H, I]
    torch::Tensor weight_hh_reverse, // [4*H, H]
    torch::Tensor bias_reverse, // [4*H]
    torch::Tensor old_h, // [D, N, H]
    torch::Tensor old_c, // [D, N, H]
    bool bidirectional = false
  ) {
  
  return lstm_cuda_forward(input,
                           weight_ih,
                           weight_hh,
                           bias,
                           weight_ih_reverse,
                           weight_hh_reverse,
                           bias_reverse,
                           old_h,
                           old_c,
                           bidirectional);
}

std::vector<torch::Tensor> lstm_cuda_backward(
    torch::Tensor grad_output, // [N, L, D*H]
    torch::Tensor grad_h, // [D, N, H]
    torch::Tensor grad_c, // [D, N, H]
    torch::Tensor weight_ih, // [4*H, I]
    torch::Tensor weight_hh, // [4*H, H]
    torch::Tensor weight_ih_reverse, // [4*H, I]
    torch::Tensor weight_hh_reverse, // [4*H, H]
    torch::Tensor i_actv, // [N, L, I]
    torch::Tensor h_actv, // [L, N, H]
    torch::Tensor c_actv, // [L, N, H]
    torch::Tensor gate_actv,  // [L, N, 4*H]
    torch::Tensor h_actv_reverse, // [L, N, H]
    torch::Tensor c_actv_reverse, // [L, N, H]
    torch::Tensor gate_actv_reverse,  // [L, N, 4*H]
    bool bidirectional = false
  );

std::vector<torch::Tensor> lstm_backward(
    torch::Tensor grad_output, // [N, L, D*H]
    torch::Tensor grad_h, // [D, N, H]
    torch::Tensor grad_c, // [D, N, H]
    torch::Tensor weight_ih, // [4*H, I]
    torch::Tensor weight_hh, // [4*H, H]
    torch::Tensor weight_ih_reverse, // [4*H, I]
    torch::Tensor weight_hh_reverse, // [4*H, H]
    torch::Tensor i_actv, // [N, L, I]
    torch::Tensor h_actv, // [L, N, H*D]
    torch::Tensor c_actv, // [L, N, H*D]
    torch::Tensor gate_actv, // [L, N, 4*H*D]
    torch::Tensor h_actv_reverse, // [L, N, H*D]
    torch::Tensor c_actv_reverse, // [L, N, H*D]
    torch::Tensor gate_actv_reverse, // [L, N, 4*H*D]
    bool bidirectional = false
  ) {
  
  return lstm_cuda_backward(grad_output,
                            grad_h,
                            grad_c,
                            weight_ih,
                            weight_hh,
                            weight_ih_reverse,
                            weight_hh_reverse,
                            i_actv,
                            h_actv,
                            c_actv,
                            gate_actv,
                            h_actv_reverse,
                            c_actv_reverse,
                            gate_actv_reverse,
                            bidirectional);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lstm_forward", &lstm_forward, "LSTM forward (CUDA)");
  m.def("lstm_backward", &lstm_backward, "LSTM backward (CUDA)");
}