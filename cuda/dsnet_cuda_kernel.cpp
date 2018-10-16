#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <cassert>

#define EPS 1e-5

namespace {
// template <typename scalar_t>
// __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
//   return 1.0 / (1.0 + exp(-z));
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
//   const auto s = sigmoid(z);
//   return (1.0 - s) * s;
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
//   const auto t = tanh(z);
//   return 1 - (t * t);
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
//   return fmax(scalar_t(0.0), z) + fmin(scalar_t(0.0), alpha * (exp(z) - scalar_t(1.0)));
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
//   const auto e = exp(z);
//   const auto d_relu = z < 0.0 ? 0.0 : 1.0;
//   return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
// }

template <typename scalar_t>
__global__ void dsnet_cuda_forward_kernel(
    const scalar_t* __restrict__ vertex,
    const scalar_t* __restrict__ elem,
    const scalar_t* __restrict__ density,
    scalar_t* __restrict__ ,
    scalar_t* __restrict__ input_gate,
    scalar_t* __restrict__ output_gate,
    scalar_t* __restrict__ candidate_cell,
    size_t state_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    input_gate[index] = sigmoid(gates[gates_row + column]);
    output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
    candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
    new_cell[index] =
        old_cell[index] + candidate_cell[index] * input_gate[index];
    new_h[index] = tanh(new_cell[index]) * output_gate[index];
  }
}

template <typename scalar_t>
__global__ void lltm_cuda_backward_kernel(
    scalar_t* __restrict__ d_old_cell,
    scalar_t* __restrict__ d_gates,
    const scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ grad_cell,
    const scalar_t* __restrict__ new_cell,
    const scalar_t* __restrict__ input_gate,
    const scalar_t* __restrict__ output_gate,
    const scalar_t* __restrict__ candidate_cell,
    const scalar_t* __restrict__ gate_weights,
    size_t state_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    const auto d_output_gate = tanh(new_cell[index]) * grad_h[index];
    const auto d_tanh_new_cell = output_gate[index] * grad_h[index];
    const auto d_new_cell =
        d_tanh(new_cell[index]) * d_tanh_new_cell + grad_cell[index];


    d_old_cell[index] = d_new_cell;
    const auto d_candidate_cell = input_gate[index] * d_new_cell;
    const auto d_input_gate = candidate_cell[index] * d_new_cell;


    const auto input_gate_index = gates_row + column;
    const auto output_gate_index = gates_row + state_size + column;
    const auto candidate_cell_index = gates_row + 2 * state_size + column;

    d_gates[input_gate_index] =
        d_input_gate * d_sigmoid(gate_weights[input_gate_index]);
    d_gates[output_gate_index] =
        d_output_gate * d_sigmoid(gate_weights[output_gate_index]);
    d_gates[candidate_cell_index] =
        d_candidate_cell * d_elu(gate_weights[candidate_cell_index]);
  }
}
} // namespace

std::vector<at::Tensor> dsnet_cuda_forward(
    at::Tensor V,
    at::Tensor E,
    at::Tensor D,
    std::vector<int64_t> res,
    std::vector<float> t,
    int64_t j,
    bool use_mass) {

  const auto n_dims = V.size(1);
  assert((n_dims == 2) || (n_dims == 3));  // GPU implementation not yet implemented for other dimensions
  assert(n_dims == res.size()); // consistent spacial dimensionality
  assert(E.size(0) == D.size(0)); // consistent number of elements

  // create noise matrix to add to V for robustness
  V += EPS * at::rand({V.size(0), V.size(1)});
  bool use_ghost = (E.size(1) == j) && (n_dims == j);
  assert((E.size(1) == j+1) || use_ghost);
  if (use_ghost){
    V = at::cat({V, at::zeros({1, V.size(1)})}, /*dim=*/0);
    E = at::cat({E, V.size(0)-1+at::zeros({E.size(0), 1}, int64_t)}, 1);
  }
  n_elem = E.size(0);
  n_vert = V.size(0);
  n_channel = D.shape(1);

  // normalize frequencies
  const double PI = std::atan(1.0)*4;
  std::vector<float> omega;
  for(auto it = t.begin(); it != t.end(); ++it) {
    omega.push_back(2 * PI / *it);
  }

  // initialize output F
  if (n_dims == 2) {
    auto F = aten::zeros({res[0], int(res[1]/2)}, /*dtype=*/std::complex<float>);
  }
  else { // n_dims == 3
    auto F = aten::zeros({res[0], res[1], int(res[2]/2)}, /*dtype=*/std::complex<float>)
  }

  // compute content array and P array
  C = at::zeros(n_elem)
  P = at::zeros({n_elem, j+1, n_dims}, float)

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.data<scalar_t>(),
        old_cell.data<scalar_t>(),
        new_h.data<scalar_t>(),
        new_cell.data<scalar_t>(),
        input_gate.data<scalar_t>(),
        output_gate.data<scalar_t>(),
        candidate_cell.data<scalar_t>(),
        state_size);
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

std::vector<at::Tensor> lltm_cuda_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights) {
  auto d_old_cell = at::zeros_like(new_cell);
  auto d_gates = at::zeros_like(gate_weights);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_old_cell.data<scalar_t>(),
        d_gates.data<scalar_t>(),
        grad_h.data<scalar_t>(),
        grad_cell.data<scalar_t>(),
        new_cell.data<scalar_t>(),
        input_gate.data<scalar_t>(),
        output_gate.data<scalar_t>(),
        candidate_cell.data<scalar_t>(),
        gate_weights.data<scalar_t>(),
        state_size);
  }));

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
