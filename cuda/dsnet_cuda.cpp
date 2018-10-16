#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> dsnet_cuda_forward(
    at::Tensor V,
    at::Tensor E,
    at::Tensor D,
    std::vector<int64_t> res,
    std::vector<float> t,
    int64_t j,
    bool use_mass
    );

std::vector<at::Tensor> lltm_cuda_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> dsnet_forward(
    at::Tensor V,
    at::Tensor E,
    at::Tensor D,
    std::vector<int64_t> res,
    std::vector<float> t,
    int64_t j,
    bool use_mass) {
  CHECK_INPUT(V);
  CHECK_INPUT(E);
  CHECK_INPUT(D);
  CHECK_INPUT(res);
  CHECK_INPUT(t);

  return lltm_cuda_forward(V, E, D, res, t, j, use_mass);
}

std::vector<at::Tensor> lltm_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);

  return lltm_cuda_backward(
      grad_h,
      grad_cell,
      new_cell,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DSNet_forward, "DSNet forward (CUDA)");
  m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}