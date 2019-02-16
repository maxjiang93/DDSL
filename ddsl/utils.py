import torch
import numpy as np
from math import factorial
import os


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

def construct_B(V, E):
    """
    Construct B matrix for Cayley-Menger Determinant
    :param V: vertex tensor. float tensor of shape (n_vertex, n_dims)
    :param E: element tensor. int tensor of shape (n_elem, j+1)
    :return: B: B matrix of shape (n_elem, j+2, j+2)
    """
    ne = E.shape[0]
    j = E.shape[1]-1
    P = V[E]
    
    B = torch.zeros(ne, j+2, j+2, device=V.device, dtype=V.dtype)
    B[:, :, 0] = 1
    B[:, 0, :] = 1
    for r in range(1, j+2):
        for c in range(r+1, j+2):
            B[:, r, c] = torch.sum((P[:, r-1] - P[:, c-1]) ** 2, dim=-1)
            B[:, c, r] = B[:, r, c]
    B[:, 0, 0] = 0
    
    return B

def construct_B_batch(P):
    """
    Construct B matrix for Cayley-Menger Determinant from P (i.e., V[E]) matrix
    :param P: point tensor. float tensor of shape (n_elem, j+1, n_dims)
    :return: B: B matrix of shape (n_elem, j+2, j+2)
    """
    ne = P.shape[0]
    j = P.shape[1]-1
    
    B = torch.zeros(ne, j+2, j+2, device=P.device, dtype=P.dtype)
    B[:, :, 0] = 1
    B[:, 0, :] = 1
    for r in range(1, j+2):
        for c in range(r+1, j+2):
            B[:, r, c] = torch.sum((P[:, r-1] - P[:, c-1]) ** 2, dim=-1)
            B[:, c, r] = B[:, r, c]
    B[:, 0, 0] = 0
    
    return B

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
        B = construct_B(V, E) # construct Cayley-Menger matrix
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
        update_val_coalesced = torch.sparse.FloatTensor(ind, val, output_shape).coalesce().to_dense()
    elif update_val.dtype is torch.float64:
        update_val_coalesced = torch.sparse.DoubleTensor(ind, val, output_shape).coalesce().to_dense()
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

def img(x, deg=1): # imaginary of tensor (assume last dim: real/imag)
    """
    multiply tensor x by i ** deg
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res

def get_polygon_E_D(V):
    """
    get E and D matrices from polygon assuming single polygon and continuous connection
    :param V: [#V, 2] 
    """
    E0 = torch.LongTensor(permute_seq(0, V.shape[0]))
    E1 = torch.LongTensor(permute_seq(1, V.shape[0]))
    E = torch.stack((E0, E1), dim=-1)
    D = torch.ones(E.shape[0], 1, device=V.device, dtype=V.dtype)
    return E, D

def normalize_V(V, margin=0.2):
    """
    normalize V into (0,1)
    :param V: [#V, 2]
    """
    V = V.clone()
    # normalize V
    V_bb = torch.max(V, dim=0)[0] - torch.min(V, dim=0)[0]
    V_c = (torch.max(V, dim=0)[0] + torch.min(V, dim=0)[0]) / 2
    V -= V_c
    V /= (1/(1-margin))*V_bb.max()
    V += 0.5
    return V

def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=torch.float64) # [dim0, dim1, dim2, d]
    dis = torch.sqrt(torch.sum(omega ** 2, dim=-1))
    filter_ = torch.exp(-0.5*((sig*2*dis/res[0])**2)).unsqueeze(-1).unsqueeze(-1)
    filter_.requires_grad = False
    return filter_

def readOBJ(filename, check=True):
    assert(os.path.exists(filename) and filename[-4:] == '.obj')
    f = open(filename, 'r')
    lines = f.readlines()
    V = []
    F = []
    for l in lines:
        llist = l.rstrip('\n').split(' ')
        if llist[0] in ['v', 'V']:
            V.append([float(llist[1]), float(llist[2]), float(llist[3])])
        elif llist[0] in ['f', 'F']:
            if check:
                if llist[1] != llist[2] and llist[1] != llist[3]:
                    F.append([int(llist[1]), int(llist[2]), int(llist[3])])
            else:
                F.append([int(llist[1]), int(llist[2]), int(llist[3])])
    V = np.array(V)
    F = np.array(F)-1
    return V, F
