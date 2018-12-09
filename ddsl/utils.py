import torch
import numpy as np


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

def simplex_content(V, E):
    """
    Compute the content of simplices in a simplicial complex
    :param V: vertex tensor. float tensor of shape (n_vertex, n_dims)
    :param E: element tensor. int tensor of shape (n_elem, j+1)
    """
    ne = E.shape[0] # number of elements
    nppe = E.shape[1] # number of points per element
    assert(nppe in [1, 2, 3, 4]) # points, lines, tri or tet
    if nppe == 1: # points
        return torch.ones(ne, 1, dtype=V.dtype)
    if nppe == 2: # lines
        P = V[E]
        Len = torch.norm(P[:, 1:] - P[:, :-1], dim=-1)
        return Len
    elif nppe == 3: # triangles
        E_ = torch.cat([E, E[:, 0:1]], dim=-1)
        P = V[E_]
        L = torch.norm(P[:, 1:] - P[:, :-1], dim=-1)
        S = torch.sum(L, dim=-1, keepdim=True) / 2
        Area = torch.sqrt(S*(S-L[:, 0:1])*(S-L[:, 1:2])*(S-L[:, 2:])) # Heron's Formula
        return Area
    elif nppe == 4: # tetrahedron
        P = V[E]
        Va = P[:, 1] - P[:, 0]
        Vb = P[:, 2] - P[:, 0]
        Vc = P[:, 3] - P[:, 0]
        Vol = torch.abs(torch.einsum('ab,ab->a', (Va, torch.cross(Vb, Vc, dim=-1))) / 6)
        return Vol

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