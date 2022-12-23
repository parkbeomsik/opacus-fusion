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
#include "grad_example_module_linear.h"
#include "quantize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Conv2dConfig>(m, "Conv2dConfig").def(py::init<int &, int &, int &, int &,
                                                           int &, int &, int &,
                                                           int &, int &, 
                                                           int &, int &, int &, int &, int &, int &, int&>());
  py::class_<LinearConfig>(m, "LinearConfig").def(py::init<int &, int &, int &, int &, int&>());                                                           
  py::class_<ReturnType>(m, "ReturnType").def(py::init<std::vector<std::vector<torch::Tensor>> &,
                                                        float &,
                                                        float &,
                                                        float &,
                                                        float &,
                                                        size_t &,
                                                        size_t &>())
                                          .def("get_per_batch_grads", &ReturnType::get_per_batch_grads)
                                          .def("get_backward_weight_ms", &ReturnType::get_backward_weight_ms)
                                          .def("get_norm_ms", &ReturnType::get_norm_ms)
                                          .def("get_clip_reduce_ms", &ReturnType::get_clip_reduce_ms)
                                          .def("get_add_noise_ms", &ReturnType::get_add_noise_ms)
                                          .def("get_workspace_size", &ReturnType::get_workspace_size)
                                          .def("get_per_example_gradient_size", &ReturnType::get_per_example_gradient_size);
  m.def("get_clip_and_reduced_grads_conv", &get_clip_and_reduced_grads_conv, "LLTM forward (CUDA)");
  m.def("get_clip_and_reduced_grads_linear", &get_clip_and_reduced_grads_linear, "LLTM forward (CUDA)");
  m.def("quantize_int8", &quantize_int8, "Quantize Int8 (CUDA)");
}