import torch
import numpy as np
from math import factorial


def fftfreqs(res, dtype=torch.float32, exact=True):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :return:
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)

    return omega

# def simplex_content(V, E):
#     """
#     Compute the content of simplices in a simplicial complex
#     :param V: vertex tensor. float tensor of shape (n_vertex, n_dims)
#     :param E: element tensor. int tensor of shape (n_elem, j+1)
#     """
#     ne = E.shape[0] # number of elements
#     nppe = E.shape[1] # number of points per element
#     assert(nppe in [1, 2, 3, 4]) # points, lines, tri or tet
#     if nppe == 1: # points
#         return torch.ones(ne, 1, dtype=V.dtype)
#     if nppe == 2: # lines
#         P = V[E]
#         Len = torch.norm(P[:, 1:] - P[:, :-1], dim=-1)
#         return Len
#     elif nppe == 3: # triangles
#         E_ = torch.cat([E, E[:, 0:1]], dim=-1)
#         P = V[E_]
#         L = torch.norm(P[:, 1:] - P[:, :-1], dim=-1)
#         S = torch.sum(L, dim=-1, keepdim=True) / 2
#         Area = torch.sqrt(S*(S-L[:, 0:1])*(S-L[:, 1:2])*(S-L[:, 2:])) # Heron's Formula
#         return Area
#     elif nppe == 4: # tetrahedron
#         P = V[E]
#         Va = P[:, 1] - P[:, 0]
#         Vb = P[:, 2] - P[:, 0]
#         Vc = P[:, 3] - P[:, 0]
#         Vol = torch.abs(torch.einsum('ab,ab->a', (Va, torch.cross(Vb, Vc, dim=-1))) / 6).unsqueeze(dim=-1)
#         return Vol

def simplex_content(V, E, signed=False):
    """
    Compute the content of simplices in a simplicial complex
    :param V: vertex tensor. float tensor of shape (n_vertex, n_dims)
    :param E: element tensor. int tensor of shape (n_elem, j+1)
    :param signed: bool denoting whether to calculate signed content or unsigned content
    :return: vol: volume of the simplex
    """
    ne = E.shape[0]
    nd = V.shape[1]
    j = E.shape[1]-1
    P = V[E]
    if not signed:
        # construct Cayley-Menger matrix
        B = torch.zeros(ne, j+2, j+2, device=V.device, dtype=V.dtype)
        B[:, :, 0] = 1
        B[:, 0, :] = 1
        for r in range(1, j+2):
            for c in range(r+1, j+2):
                B[:, r, c] = torch.sum((P[:, r-1] - P[:, c-1]) ** 2, dim=-1)
                B[:, c, r] = B[:, r, c]
        B[:, 0, 0] = 0
        vol2 = (-1)**(j+1) / (2**j) / (factorial(j)**2) * batch_det(B)
        neg_mask = torch.sum(vol2 < 0)
        if torch.sum(neg_mask) > 0:
            vol2[neg_mask] = 0
            print("[!]Warning: zeroing {0} small negative number".format(torch.sum(neg_mask).item()))
        vol =torch.sqrt(vol2)
    else:
        assert(j == nd)
        # matrix determinant
        mat = P[:, :-1] - P[:, -1:]
        vol = batch_det(mat) / factorial(j)

    return vol.unsqueeze(-1)

def triangulate_interior(V, E):
    """
    Triangulate/tetrahedronize interior of shape given boundary simplex mesh
    :param V: vertex tensor. float tensor of shape (n_vertex, n_dims)
    :param E: element tensor. int tensor of shape (n_elem, j)
    :return: new_E: new element matrix of shape (n_elem, j+1)
    """
    return new_E

def permute_seq(i, len):
    """
    Permute the ordering of integer sequences
    """
    assert(i<len)
    return list(range(i, len)) + list(range(0, i))

def coalesce_update(update_ind, update_val, output_shape):
    """
    Coalesce update values for batched scenario
    :param update_ind: indices for batched Ei matrix
    :param update_val: update values to be massaged into output shape
    :param output_shape: desired output shape
    """
    dev = update_val.device
    b = update_ind.shape[0]
    n_dims = output_shape[1]
    n_pts = len(update_ind.view(-1))

    rid = update_ind.unsqueeze(-1).repeat(1,1,n_dims).view(-1)
    cid = torch.LongTensor(list(range(n_dims)) * n_pts).to(dev)
    ind = torch.stack([rid, cid], dim=0)
    val = update_val.view(-1)
    if update_val.dtype is torch.float32:
        update_val_coalesced = torch.sparse.FloatTensor(ind, val, output_shape, device=dev).coalesce().to_dense()
    elif update_val.dtype is torch.float64:
        update_val_coalesced = torch.sparse.DoubleTensor(ind, val, output_shape, device=dev).coalesce().to_dense()
    else:
        print("ERROR: Unsupported data type {}, must be torch.float32 or torch.float64.".format(update_val.dtype))
    return update_val_coalesced

def batch_det(A):
    """
    Batch determinant of square matrix A of shape (*, N, N)
    Return:
    Tensor of shape (*)
    """
    LU, pivots = A.btrifact()
    det_LU = torch.einsum('...ii->...i', LU).prod(-1)
    pivots -= 1
    d = pivots.shape[-1]
    perm = pivots - torch.arange(d, dtype=pivots.dtype, device=pivots.device).expand(pivots.shape)
    det_P = (-1) ** ((perm != 0).sum(-1))
    det = det_LU * det_P.type(det_LU.dtype)

    return det

def batch_cofactor(A):
    """
    Batch cofactor matrix of square matrix A of shape (*, N, N)
    Return:
    Batched cofactor matrix (tensor) of shape (*, N, N) 
    """
    A_inv = torch.inverse(A)
    A_det = batch_det(A).unsqueeze(-1).unsqueeze(-1)

    return torch.transpose(A_inv, -1, -2) * A_det

def batch_adjugate(A):
    """
    Batch adjugate matrix of square matrix A of shape (*, N, N)
    Return:
    Batched adjugate matrix (tensor) of shape (*, N, N) 
    """
    A_inv = torch.inverse(A)
    A_det = batch_det(A).unsqueeze(-1).unsqueeze(-1)

    return A_inv * A_det

