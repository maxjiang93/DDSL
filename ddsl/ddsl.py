import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from math import pi

from utils import fftfreqs, simplex_content, permute_seq, coalesce_update, img, construct_B_batch, batch_adjugate
from math import ceil, factorial

class SimplexFT(Function):
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
        ## check if E is subdim
        subdim = E.shape[1] == j and V.shape[1] == j
        assert (E.shape[1] == j+1 or subdim)
        if subdim:
            assert((D == D[0]).sum().item() == D.numel()) # assert same densities for all simplices (homogeneous filling)
            V = torch.cat((V, torch.zeros(1, V.shape[1], device=V.device, dtype=V.dtype)), dim=0)
            E = torch.cat((E, V.shape[0] - 1 + torch.zeros((E.shape[0], 1), device=E.device, dtype=E.dtype)), dim=1)
        n_elem = E.shape[0]
        n_vert = V.shape[0]
        n_channel = D.shape[1]

        ## save context info for backwards
        ctx.mark_non_differentiable(E, D) # mark non-differentiable
        ctx.res = res
        ctx.t = t
        ctx.j = j
        ctx.mode = mode
        ctx.n_dims = V.shape[1]
        ctx.elem_batch = elem_batch
        ctx.subdim = subdim
        
        # compute content array
        C = factorial(j) * simplex_content(V, E, signed=subdim) # [n_elem, 1]
        ctx.save_for_backward(V, E, D, C)

        ## compute frequencies F
        n_dims = ctx.n_dims
        assert(n_dims == len(res))  # consistent spacial dimensionality
        assert(E.shape[0] == D.shape[0])  # consistent vertex numbers
        assert(mode in ['density', 'mass'])
        
        # frequency tensor
        omega = fftfreqs(res, dtype=V.dtype).to(V.device) # [dim0, dim1, dim2, d]

        # normalize frequencies
        for dim in range(n_dims):
            omega[..., dim] *= 2 * pi / t[dim]

        # initialize output F
        F_shape = list(omega.shape)[:-1]
        F_shape += [n_channel, 2]
        F = torch.zeros(*F_shape, dtype=V.dtype, device=V.device) # [dimX, dimY, dimZ, n_chan, 2] 2: real/imag

        # compute element-point tensor
        P = V[E] # [n_elem, j+1, d]

        # loop over element batches
        for idx in range(ceil(n_elem/elem_batch)):
            id_start = idx * elem_batch
            id_end = min((idx+1) * elem_batch, n_elem)
            Xi = P[id_start:id_end] # [elem_batch, j+1, d]
            Di = D[id_start:id_end] # [elem_batch, n_channel]
            Ci = C[id_start:id_end] # [elem_batch, 1]
            CDi = Ci * Di # [elem_batch, n_channel]
            sig = torch.einsum('bjd,...d->bj...', (Xi, omega)) 
            sig = torch.unsqueeze(sig, dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1]
            esig = torch.stack((torch.cos(sig), -torch.sin(sig)), dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 2]
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
            F += Fi

        F = img(F, deg=j) # Fi *= 1j**j [dimX, dimY, dimZ, n_chan, 2] 2: real/imag

        if mode == 'density':
            res_t = torch.tensor(res)
            if not torch.equal(res_t, res[0]*torch.ones(len(res), dtype=res_t.dtype)):
                print("WARNING: density preserving mode not correctly implemented if not all res are equal")
            F *= res[0] ** j
        return F

    @staticmethod
    @once_differentiable
    def backward(ctx, dF):
        """
        :param dF: per-frequency sensitivity from downstream layers of shape [dimX, dimY, dimZ, n_channel, 2]
        """
        if ctx.needs_input_grad[0]:
            V, E, D, C = ctx.saved_tensors

            n_dims = V.shape[1]
            n_elem = E.shape[0]
            n_vert = V.shape[0]
            n_channel = D.shape[1]

            # recover context
            res = ctx.res
            t = ctx.t
            j = ctx.j
            mode = ctx.mode
            n_dims = ctx.n_dims
            elem_batch = ctx.elem_batch

            # frequency tensor
            omega = fftfreqs(res, dtype=V.dtype).to(V.device) # [dim0, dim1, dim2, d]

            # normalize frequencies
            for dim in range(n_dims):
                omega[..., dim] *= 2 * pi / t[dim]

            # compute element-point tensor
            P = V[E] # [n_elem, j+1, d]

            # initialize output dV
            dV = torch.zeros_like(V) # [j+1, n_dims]

            # compute element-point tensor
            P = V[E] # [n_elem, j+1, n_dims]

            # helper functions
            seq = lambda i : permute_seq(i, j+1) # return looped sequences

            # loop over element batches
            for idx in range(ceil(n_elem/elem_batch)):
                id_start = idx * elem_batch
                id_end = min((idx+1) * elem_batch, n_elem)
                elem_batch_i = id_end - id_start
                Xi = P[id_start:id_end] # [elem_batch, j+1, n_dims]
                Di = D[id_start:id_end] # [elem_batch, n_channel]
                Ci = C[id_start:id_end] # [elem_batch, 1]
                Ei = E[id_start:id_end] # [elem_batch, j+1]
                sig = torch.einsum('bjd,...d->bj...', (Xi, omega)) 
                sig = torch.unsqueeze(sig, dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1]
                esig = torch.stack((torch.cos(sig), -torch.sin(sig)), dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 2]
                sig = torch.unsqueeze(sig, dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 1]
                denom = torch.ones_like(sig) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 1]
                for dim in range(1, j+1):
                    denom *= sig - sig[:, seq(dim)]
                Si = esig / denom # [elem_batch, j+1, dimX, dimY, dimZ, 1, 2]
                # first part: tmp
                tmp = torch.zeros_like(Si) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 2]
                for dim in range(1, j+1):
                    tmp += (Si[:, seq(dim)] + Si) / (sig[:, seq(dim)] - sig)
                tmp -= img(Si, deg=1) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 2]
                tmp = tmp.permute(*(list(range(2, 2+n_dims)) + [0, 1] + list(range(2+n_dims, len(tmp.shape)))))
                tmp[tuple([0] * n_dims)] = 0
                tmp = tmp.permute(*(list(range(n_dims, 2+n_dims)) + list(range(n_dims)) + list(range(2+n_dims, len(tmp.shape)))))
                tmp = img(tmp, deg=j)
                tmp *= dF # [elem_batch, j+1, dimX, dimY, dimZ, n_channel, 2]
                tmp = tmp.unsqueeze(-3) * omega.unsqueeze(-1).unsqueeze(-1) # [elem_batch, j+1, dimX, dimY, dimZ, n_dims, n_channel, 2]
                tmp = torch.sum(tmp, dim=tuple([-1]+list(range(2, 2+n_dims)))) # [elem_batch, j+1, n_dims, n_channel]
                tmp = torch.sum(tmp * Di.unsqueeze(1).unsqueeze(1), dim=-1)     # [elem_batch, j+1, n_dims]
                tmp *= Ci.unsqueeze(-1) # [elem_batch, j+1, n_dims]
                
                # second part: tmp2
                S = torch.sum(Si, dim=1) # [elem_batch, dimX, dimY, dimZ, 1, 2]
                S = S.permute(*(list(range(1, 1+n_dims)) + [0] + list(range(1+n_dims, len(S.shape))))) # [dimX, dimY, dimZ, elem_batch, 1, 2]
                S[tuple([0] * n_dims)] = - 1 / factorial(j)
                S = S.permute(*([n_dims] + list(range(n_dims)) + list(range(1+n_dims, len(S.shape))))) # [elem_batch, dimX, dimY, dimZ, 1, 2]
                tmp2 = img(S, deg=j) * dF # [elem_batch, dimX, dimY, dimZ, n_channel, 2]
                tmp2 = torch.sum(tmp2, dim=tuple([-1] + list(range(1, 1+n_dims)))) # [elem_batch, n_channel]
                tmp2 *= ((-1)**(j+1)) / (2**j) 
                tmp2 /= Ci # [elem_batch, n_channel]
                tmp2 = (tmp2 * Di).sum(-1) # [elem_batch]
                # summation term
                B = construct_B_batch(Xi) # [elem_batch, j+2, j+2]
                B_adj = batch_adjugate(B) # [elem_batch, j+2, j+2]
                B_adj_sub = B_adj[:, 1:, 1:] # [elem_batch, j+1, j+1]
                tmpsum = torch.zeros_like(Xi) # [elem_batch, j+1, n_dims]
                rind = list(range(j+1))
                for dim in range(1, j+1):
                    adj = B_adj_sub[:, rind, seq(dim)].unsqueeze(-1) # [elem_batch, j+1, 1]
                    tmpsum += 2 * (Xi - Xi[:, seq(dim)]) * adj # [elem_batch, j+1, n_dims]
                tmp2 = tmpsum * tmp2.unsqueeze(-1).unsqueeze(-1)  # [elem_batch, j+1, n_dims]
                
                tmp += tmp2
                ddV = coalesce_update(Ei, tmp, dV.shape)
                dV += ddV
                
            if mode == "density":
                dV *= res[0] ** j
                
            if ctx.subdim:
                dV = dV[:-1]
                
        else:
            dV = None
            
        return dV, None, None, None, None, None, None, None

    
class DDSL_spec(nn.Module):
    def __init__(self, res, t, j, elem_batch=100, mode='density'):
        super(DDSL_spec, self).__init__()
        self.res = res
        self.t = t
        self.j = j
        self.elem_batch = elem_batch
        self.mode = mode
    def forward(self, V, E, D):
        return SimplexFT.apply(V,E,D,self.res,self.t,self.j,self.elem_batch,self.mode)