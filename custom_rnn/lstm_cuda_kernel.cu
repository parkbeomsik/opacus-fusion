#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

template <typename scalar_t>
__global__ void lstm_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_c,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_c,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> new_gates) {
  // batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
    scalar_t i_t = sigmoid(gates[n][0][c]);
    scalar_t f_t = sigmoid(gates[n][1][c]);
    scalar_t g_t = tanh   (gates[n][2][c]);
    scalar_t o_t = sigmoid(gates[n][3][c]);

    new_c[n][c] = f_t * old_c[n][c] + i_t * g_t;
    new_h[n][c] = o_t * tanh(new_c[n][c]);

    new_gates[n][0][c] = i_t;
    new_gates[n][1][c] = f_t;
    new_gates[n][2][c] = g_t;
    new_gates[n][3][c] = o_t;
  }
}

std::vector<torch::Tensor> lstm_cuda_forward(
    torch::Tensor input, // [N, L, I]
    torch::Tensor weight_ih, // [4*H, I]
    torch::Tensor weight_hh, // [4*H, H]
    torch::Tensor bias, // [4*H]
    torch::Tensor weight_ih_reverse, // [4*H, I]
    torch::Tensor weight_hh_reverse, // [4*H, H]
    torch::Tensor bias_reverse, // [4*H]
    torch::Tensor h_0, // [D, N, H]
    torch::Tensor c_0, // [D, N, H]
    bool bidirectional = false
  ) {
  
  using namespace torch::indexing;  

  input.transpose_(0, 1); // [L, N, I]

  int D = bidirectional ? 2 : 1;
  int N = input.size(1);
  int L = input.size(0);
  int I = input.size(2);
  int H = h_0.size(2);

  // Check tensor shapes
  assert(weight_ih.size(0) == 4 * H);
  assert(weight_ih.size(1) == I);
  assert(weight_hh.size(0) == 4 * H);
  assert(weight_hh.size(1) == H );
  assert(bias.size(1) == 4 * H);
  if (bidirectional) {
    assert(weight_ih_reverse.size(0) == 4 * H);
    assert(weight_ih_reverse.size(1) == I);
    assert(weight_hh_reverse.size(0) == 4 * H);
    assert(weight_hh_reverse.size(1) == H );
    assert(bias_reverse.size(1) == 4 * H);
  }
  assert(h_0.size(0) == D);
  assert(h_0.size(1) == N);
  assert(h_0.size(2) == H);
  assert(c_0.size(0) == D);
  assert(c_0.size(1) == N);
  assert(c_0.size(2) == H);

  // std::cout << input << std::endl;
  // std::cout << weight_ih << std::endl;

  auto xW = torch::matmul(input.reshape({input.size(0) * input.size(1), input.size(2)}), 
                          weight_ih.transpose(0, 1)); // [L*N, 4*H]
  xW = xW.reshape({L, N, 4*H}); // [L, N, 4*H]

  // std::cout << xW << std::endl;

  // Create tensors which store activations
  auto gate_actv = torch::empty({L, N, 4*H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  auto h_actv = torch::empty({L+1, N, H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  auto c_actv = torch::empty({L+1, N, H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  auto gate_actv_reverse = torch::empty({0});
  auto h_actv_reverse = torch::empty({0});
  auto c_actv_reverse = torch::empty({0});
  if (bidirectional) {
    gate_actv_reverse = torch::empty({L, N, 4*H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    h_actv_reverse = torch::empty({L+1, N, H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    c_actv_reverse = torch::empty({L+1, N, H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  }

  // Set h_0 and c_0
  h_actv.index_put_({0}, h_0.index({0}));
  c_actv.index_put_({0}, c_0.index({0}));
  // Set h_n and c_n for backward layer (only with bidrectional)
  if (bidirectional) {
    h_actv_reverse.index_put_({L}, h_0.index({1}));
    c_actv_reverse.index_put_({L}, c_0.index({1}));
  }

  // Shape of CUDA Threadblock
  const int threads = 1024;
  const dim3 blocks((H + threads - 1) / threads, N);

  // Forward layer
  for (int t=1; t<L+1; ++t) {
    auto old_h = h_actv.index({t-1}); // [N, H]
    // Compute h_(t-1) * W_hh + i_(t-1) * W_ih + bias -> [N, 4*H]
    auto gate_input = torch::addmm(bias + xW.index({t-1}),
                                   old_h,
                                   weight_hh.transpose(0, 1)); // [N, 4*H]

    // std::cout << gate_input << std::endl;
    
    auto old_c = c_actv.index({t-1}); // [N, H]
    auto new_h = torch::empty_like(old_c); // [N, H]
    auto new_c = torch::empty_like(old_c); // [N, H]
    gate_input = gate_input.view({N, 4, H}); // [N, 4, H]
    auto new_gate = torch::empty_like(gate_input); // [N, 4, H]

    // Perform LSTM gate function
    AT_DISPATCH_FLOATING_TYPES(gate_input.type(), "lstm_forward_cuda", ([&] {
      lstm_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
          gate_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          old_c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          new_c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          new_gate.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));

    // Store gate activation
    gate_actv.index_put_({t-1}, new_gate.view({N, 4*H}));

    // Set next h and c
    h_actv.index_put_({t}, new_h);
    c_actv.index_put_({t}, new_c);
  }
  auto output = h_actv.index({Slice(1, None)}); // [L, N, H]

  if (bidirectional) {
    auto xW_reverse = torch::matmul(input.reshape({input.size(0) * input.size(1), input.size(2)}), 
                            weight_ih_reverse.transpose(0, 1)); // [L*N, 4*H]
    xW_reverse = xW.reshape({L, N, 4*H}); // [L, N, 4*H]

    // Backward layer
    for (int t=L-1; t>=0; --t) {
      auto old_h = h_actv_reverse.index({t+1}); // [N, H]
      // Compute h_(t+1) * W_hh + i_(t+1) * W_ih + bias -> [N, 4*H]
      auto gate_input = torch::addmm(bias + xW_reverse.index({t}),
                                     old_h,
                                     weight_hh_reverse.transpose(0, 1)); // [N, 4*H]
      
      auto old_c = c_actv_reverse.index({t+1}); // [N, H]
      auto new_h = torch::empty_like(old_c); // [N, H]
      auto new_c = torch::empty_like(old_c); // [N, H]
      gate_input = gate_input.view({N, 4, H}); // [N, 4, H]
      auto new_gate = torch::empty_like(gate_input); // [N, 4, H]

      // Perform LSTM gate function
      AT_DISPATCH_FLOATING_TYPES(gate_input.type(), "lstm_forward_cuda", ([&] {
        lstm_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            gate_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            old_c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            new_c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            new_gate.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
      }));

      // Store gate activation
      gate_actv_reverse.index_put_({t}, new_gate.view({N, 4*H}));

      // Set next h and c
      h_actv_reverse.index_put_({t}, new_h);
      c_actv_reverse.index_put_({t}, new_c);
    }

    output = torch::cat({output, h_actv_reverse.index({Slice(0, L)})}, 2);  // [L, N, D*H]
  }

  input.transpose_(0, 1); // [N, L, I]
  output.transpose_(0, 1); // [N, L, D*H]

  // Set output h_n and c_n
  auto h_n = torch::empty_like(h_0);
  h_n.index_put_({0}, h_actv.index({L}));
  if (bidirectional) {
    h_n.index_put_({1}, h_actv_reverse.index({0}));
  }
  auto c_n = torch::empty_like(c_0);
  c_n.index_put_({0}, c_actv.index({L}));
  if (bidirectional) {
    c_n.index_put_({1}, c_actv_reverse.index({0}));
  }

  if (bidirectional) {
    h_actv_reverse = h_actv_reverse.index({Slice(1, None)});
    c_actv_reverse = c_actv_reverse.index({Slice(1, None)});
  }

  return {output, h_n, c_n, input, 
          h_actv.index({Slice(0, L)}), c_actv.index({Slice(0, L)}), gate_actv,
          h_actv_reverse, c_actv_reverse, gate_actv_reverse};
}

template <typename scalar_t>
__global__ void lstm_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> new_grad_gates,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_grad_c,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_c,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_c) {
  // batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
    const auto i_t = gates[n][0][c];
    const auto f_t = gates[n][1][c];
    const auto g_t = gates[n][2][c];
    const auto o_t = gates[n][3][c];

    const auto c_t = old_c[n][c] * f_t + i_t * g_t;

    const auto d_c_t = grad_c[n][c] + d_tanh(c_t) * o_t * grad_h[n][c];

    auto d_f_t = old_c[n][c] * d_c_t;
    auto d_i_t = g_t * d_c_t;
    auto d_g_t = i_t * d_c_t;
    auto d_o_t = tanh(c_t) * grad_h[n][c];

    new_grad_gates[n][0][c] = (1 - i_t) * i_t * d_i_t; 
    new_grad_gates[n][1][c] = (1 - f_t) * f_t * d_f_t;
    new_grad_gates[n][2][c] = (1 - g_t * g_t) * d_g_t;
    new_grad_gates[n][3][c] = (1 - o_t) * o_t * d_o_t;

    new_grad_c[n][c] = f_t * d_c_t;
  }
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
  ) {

  using namespace torch::indexing;  

  grad_output.transpose_(0, 1); // [L, N, D*H]
  i_actv.transpose_(0, 1); // [L, N, I]
  
  int D = bidirectional ? 2 : 1;
  int N = i_actv.size(1);
  int L = i_actv.size(0);
  int I = i_actv.size(2);
  int H = grad_h.size(2);

  // Check tensor shapes
  assert(weight_ih.size(0) == 4 * H);
  assert(weight_ih.size(1) == I);
  assert(weight_hh.size(0) == 4 * H);
  assert(weight_hh.size(1) == H);
  if (bidirectional) {
    assert(weight_ih_reverse.size(0) == 4 * H);
    assert(weight_ih_reverse.size(1) == I);
    assert(weight_hh_reverse.size(0) == 4 * H);
    assert(weight_hh_reverse.size(1) == H);
  }
  assert(grad_output.size(0) == L);
  assert(grad_output.size(1) == N);
  assert(grad_output.size(2) == D*H);
  assert(grad_h.size(0) == D);
  assert(grad_h.size(1) == N);
  assert(grad_h.size(2) == H);
  assert(grad_c.size(0) == D);
  assert(grad_c.size(1) == N);
  assert(grad_c.size(2) == H);
  assert(h_actv.size(0) == L);
  assert(h_actv.size(1) == N);
  assert(h_actv.size(2) == H);
  assert(c_actv.size(0) == L);
  assert(c_actv.size(1) == N);
  assert(c_actv.size(2) == H);
  assert(gate_actv.size(0) == L);
  assert(gate_actv.size(1) == N);
  assert(gate_actv.size(2) == 4*H);
  if (bidirectional) {
    assert(h_actv_reverse.size(0) == L);
    assert(h_actv_reverse.size(1) == N);
    assert(h_actv_reverse.size(2) == H);
    assert(c_actv_reverse.size(0) == L);
    assert(c_actv_reverse.size(1) == N);
    assert(c_actv_reverse.size(2) == H);    
    assert(gate_actv_reverse.size(0) == L);
    assert(gate_actv_reverse.size(1) == N);
    assert(gate_actv_reverse.size(2) == 4*H);
  }

  const int threads = 1024;
  const dim3 blocks((H + threads - 1) / threads, N);

  // Create tensors to store return grads
  auto total_grad_gates = torch::empty({L, N, 4*H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  auto total_grad_gates_reverse = torch::empty({0});
  if (bidirectional) {
    total_grad_gates_reverse = torch::empty({L, N, 4*H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  }
  auto grad_h_0 = torch::empty({D, N, H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  auto grad_c_0 = torch::empty({D, N, H}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  // Forward layer
  auto old_grad_h = grad_h.index({0}); // [N, H]
  auto old_grad_c = grad_c.index({0}); // [N, H]
  for (int t=L-1; t >= 0; --t) {
    // Set grad_h at current timestamp
    old_grad_h += grad_output.index({t, Slice(), Slice(None, H)}); // [N, H]

    auto new_grad_c = torch::empty_like(old_grad_c); // [N, H]
    auto gates = gate_actv.index({t}); // [N, 4*H]
    gates = gates.view({N, 4, H}); // [N, 4, H]
    auto new_grad_gates = torch::empty_like(gates); // [N, 4, H]
    auto c = c_actv.index({t}); // [N, H]

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lstm_backward_cuda", ([&] {
      lstm_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
          new_grad_gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          new_grad_c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          old_grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          old_grad_c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));

    new_grad_gates = new_grad_gates.view({N, 4*H}); // [N, 4*H]

    // Store grad_gates to compute weight and bias gradients
    total_grad_gates.index_put_({t}, new_grad_gates);

    // Compute grad hidden vector
    auto new_grad_h = torch::matmul(new_grad_gates,
                                    weight_hh); // [N, H]

    old_grad_h = new_grad_h;
    old_grad_c = new_grad_c;
  }
  // Compute grad input
  auto grad_input = torch::matmul(
    total_grad_gates.view({L*N, 4*H}), weight_ih); // [L*N, I]
  // Set grad_h_0 and grad_c_0
  grad_h_0.index_put_({0}, old_grad_h);
  grad_c_0.index_put_({0}, old_grad_c);

  if (bidirectional) {
    // Backward layer
    auto old_grad_h = grad_h.index({1}); // [N, H]
    auto old_grad_c = grad_c.index({1}); // [N, H]
    for (int t=0; t < L; ++t) {
      // Set grad_h at current timestamp
      old_grad_h += grad_output.index({t, Slice(), Slice(H, None)}); // [N, H]

      auto new_grad_c = torch::empty_like(old_grad_c); // [N, H]
      auto gates = gate_actv_reverse.index({t}); // [N, 4*H]
      gates = gates.view({N, 4, H}); // [N, 4, H]
      auto new_grad_gates = torch::empty_like(gates); // [N, 4, H]
      auto c = c_actv_reverse.index({t}); // [N, H]

      AT_DISPATCH_FLOATING_TYPES(gates.type(), "lstm_backward_cuda", ([&] {
        lstm_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            new_grad_gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            new_grad_c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            old_grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            old_grad_c.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
      }));

      new_grad_gates = new_grad_gates.view({N, 4*H}); // [N, 4*H]

      // Store grad_gates to compute weight and bias gradients
      total_grad_gates_reverse.index_put_({t}, new_grad_gates);

      // Compute grad hidden vector
      auto new_grad_h = torch::matmul(new_grad_gates,
                                      weight_hh_reverse); // [N, H]

      old_grad_h = new_grad_h;
      old_grad_c = new_grad_c;
    }
    // Compute grad input
    grad_input += torch::matmul(
      total_grad_gates_reverse.view({L*N, 4*H}), weight_ih_reverse);
    // Set grad_h_0 and grad_c_0
    grad_h_0.index_put_({1}, old_grad_h);
    grad_c_0.index_put_({1}, old_grad_c);
  }

  grad_input = grad_input.view({L, N, I}); // [L, N, I]
  grad_input = grad_input.transpose(0, 1); // [N, L, I]
  i_actv.transpose_(0, 1); // [N, L, I]

  grad_output.transpose_(0, 1); // [N, L, D*H]

  return {grad_input, grad_h_0, grad_c_0, total_grad_gates, total_grad_gates_reverse};
}