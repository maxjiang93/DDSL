import torch
from torch.autograd import Function

from utils import fftfreqs, simplex_content, triangulate_interior, permute_seq
from math import ceil, factorial


class DDSL(Function):
    """
    Fourier transform for signal defined on a j-simplex set in R^n space
    :param V: vertex tensor. float tensor of shape (n_vertex, n_dims)
    :param E: element tensor. int tensor of shape (n_elem, j or j+1)
              if j cols, triangulate/tetrahedronize interior first.
    :param D: int ndarray of shape (n_vertex, n_channel)
    :param res: n_dims int tuple of number of frequency modes
    :param t: n_dims tuple of period in each dimension
    :param j: dimension of simplex set
    :param mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
    :return: F: ndarray of shape (res[0], res[1], ..., res[-1]/2, n_channel)
                last dimension is halfed since the signal is assumed to be real
    """
    @staticmethod
    def forward(ctx, V, E, D, res, t, j, elem_batch=100, mode='density'):
        ## boiler-plate
        ctx.res = res
        ctx.t = t
        ctx.j = j
        ctx.mode = mode
        ctx.save_for_backward(V, E, D)

        ## compute frequencies F
        n_dims = V.shape[1]
        assert(n_dims == len(res))  # consistent spacial dimensionality
        assert(E.shape[0] == D.shape[0])  # consistent vertex numbers
        assert(mode in ['density', 'mass'])

        # number of columns in E
        subdim = E.shape[1] == j and n_dims == j
        assert (E.shape[1] == j+1 or subdim)
        if subdim:
            E = triangulate_interior(V, E)
        n_elem = E.shape[0]
        n_vert = V.shape[0]
        n_channel = D.shape[1]

        # frequency tensor
        omega = fftfreqs(res).to(V.device) # [dim0, dim1, dim2, d]
        omega[tuple([0] * n_dims)] += 1 # will get rid of this

        # normalize frequencies
        for dim in range(n_dims):
            omega[..., dim] *= 2 * np.pi / t[dim]

        # initialize output F
        F_shape = list(omega.shape)[:-1]
        F_shape += [n_channel, 2]
        F = torch.zeros(*F_shape, dtype=V.dtype, device=V.device) # [dimX, dimY, dimZ, n_chan, 2] 2: real/imag

        # compute content array
        C = simplex_content(V, E) # [n_elem, 1]

        # compute element-point tensor
        P = V[E] # [n_elem, j+1, d]

        # loop over element batches
        for idx in range(ceil(n_elem/elem_batch)):
            id_start = idx * elem_batch
            id_end = min((idx+1) * elem_batch, n_elem)
            X = P[id_start:id_end] # [elem_batch, j+1, d]
            Di = D[id_start:id_end] # [elem_batch, n_channel]
            Ci = C[id_start:id_end] # [elem_batch, 1]
            CDi = Ci * Di # [elem_batch, n_channel]
            sig = torch.einsum('bjd,...d->bj...', (X, omega)) 
            sig = torch.unsqueeze(sig, dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1]
            esig = torch.stack((torch.cos(sig), torch.sin(sig)), dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 2]
            sig = torch.unsqueeze(sig, dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 1]
            denom = torch.ones_like(sig) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 1]
            for dim in range(1, j+1):
                seq = permute_seq(dim, j+1)
                denom *= sig - sig[:, seq]
            tmp = torch.sum(esig / denom, dim=1) # [elem_batch, dimX, dimY, dimZ, 1, 2]
            CDi.unsqueeze_(-1) # [elem_batch, n_channel, 1]
            for _ in range(n_dims): # unsqueeze to broadcast
                CDi.unsqueeze_(dim=1) # [elem_batch, 1, 1, 1, n_channel, 1]
            tmp *= CDi # [elem_batch, dimX, dimY, dimZ, n_channel, 2]
            Fi = torch.sum(tmp, dim=0, keepdim=False) # [dimX, dimY, dimZ, n_channel, 2]
            Fi[tuple([0] * n_dims)] = - 1 / factorial(j) * torch.sum(CDi, dim=0).unsqueeze(dim=-1)
            Fi *= factorial(j)
            # Fi *= 1j**j # [dimX, dimY, dimZ, n_chan, 2] 2: real/imag
            F += Fi
        if j == 0:
            pass
        elif j == 1:
            F = F[..., [1, 0]]
            F[..., 0] = -F[..., 0]
        elif j == 2:
            F *= -1
        elif j == 3:
            F = F[..., [1, 0]]
            F[..., 1] = -F[..., 1]

        if mode == 'density':
            if not np.array_equal(res, res[0]*np.ones(len(res))):
                print("WARNING: density preserving mode not correctly implemented if not all res are equal")
            F *= res[0] ** j
        return F

    @staticmethod
    def backward(ctx, dF):
        V, E, D = ctx.saved_tensors
        dV = ddsl_cuda.backward(dF, V, E, D, res, t, j, mode=ctx.mode)
        
        n_dims = V.shape[1]
        assert(n_dims == 2) # backwards not implemented for other dims yet
        assert(ctx.j == 2) # not yet implemented for other simplices
        
        n_elem = E.shape[0]
        n_vert = V.shape[0]
        n_channel = D.shape[1]
        
        
        return dV, None, None, None, None, None, None