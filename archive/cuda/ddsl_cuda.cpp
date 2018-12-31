#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> ddsl_cuda_forward(
    at::Tensor V,
    at::Tensor E,
    at::Tensor D,
    std::vector res,
    std::vector t,
    int j,
    int mode);

std::vector<at::Tensor> ddsl_cuda_backward(
	at::Tensor dF,
    at::Tensor V,
    at::Tensor E,
    at::Tensor D,
    std::vector res,
    std::vector t,
    int j,
    int mode);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> ddsl_forward(
    at::Tensor V,
    at::Tensor E,
    at::Tensor D,
    std::vector res,
    std::vector t,
    int j,
    int mode) {
  CHECK_INPUT(V);
  CHECK_INPUT(E);
  CHECK_INPUT(D);

  return ddsl_cuda_forward(V, E, D, res, t, j, mode);
}

std::vector<at::Tensor> ddsl_backward(
    at::Tensor dF,
    at::Tensor V,
    at::Tensor E,
    at::Tensor D,
    std::vector res,
    std::vector t,
    int j,
    int mode) {
	CHECK_INPUT(dF);
  CHECK_INPUT(V);
  CHECK_INPUT(E);
  CHECK_INPUT(D);

  return ddsl_cuda_backward(dF, V, E, D, res, t, j, mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ddsl_forward, "DDSL forward (CUDA)");
  m.def("backward", &ddsl_backward, "DDSL backward (CUDA)");
}
