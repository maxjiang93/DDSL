import torch
from torch.autograd import Function

import ddsl_cuda

class DDSL(Function):
	"""
    Fourier transform for signal defined on a j-simplex set in R^n space
    :param V: vertex tensor. float tensor of shape (n_vertex, n_dims)
    :param E: element tensor. int tensor of shape (n_edge, j or j+1)
              if j cols, then assume sign of area using RHR and use ghost node.
    :param D: int ndarray of shape (n_vertex, n_channel)
    :param res: n_dims int tuple of number of frequency modes
    :param t: n_dims tuple of period in each dimension
    :param j: dimension of simplex set
    :param mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
    :param prog_bar: True to turn on tqdm progress bar
    :return: F: ndarray of shape (res[0], res[1], ..., res[-1]/2, n_channel)
                last dimension is halfed since the signal is assumed to be real
	"""
	@staticmethod
	def forward(ctx, V, E, D, res, t, j, mode='density'):
		mode_id = 0 if mode=='density' else 1
		F = ddsl_cuda.forward(V, E, D, res, t, j, mode=mode_id)
		ctx.res = res
		ctx.t = t
		ctx.j = j
		ctx.mode = mode_id
		ctx.save_for_backward(V, E, D)
		return F

	@staticmethod
	def backward(ctx, dF):
		V, E, D = ctx.saved_tensors
		dV = ddsl_cuda.backward(dF, V, E, D, res, t, j, mode=ctx.mode)
		return dV, None, None, None, None, None, None
