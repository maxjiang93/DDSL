import numpy as np
from collections import deque
import gc
from copy import deepcopy
from math import factorial
from cmath import exp
from tqdm import tqdm
from numba import jit, cuda
import numba as nb


def fftpad(F, res, norm=True):
    """
    Utility function to pad frequency domain with zeros to a specified resolution
    :param F: n_dims tuple of period in each dimension
    :param res: n_dims int tuple of number of desired frequency modes
    :param norm: normalize output to have same physical density as original signal
    
    :return: F_pad: padded frequencies
    """
    for rprev, rnew in zip(F.shape, res):
        assert(rnew >= rprev)
        assert((rnew - rprev) % 2 == 0)
    pad_list = []
    for i in range(len(res)):
        pad_list.append((int((res[i] - F.shape[i])/2), int((res[i] - F.shape[i])/2)))
    for i in range(len(res), len(F.shape)):
        pad_list.append((0, 0))
    F_pad = np.pad(F, tuple(pad_list), 'constant')

    if norm:
        scale = 1
        for i in range(len(res)):
            scale *= res[i] / F.shape[i]
        F_pad *= scale

    return F_pad


def rfftpad(F, res, norm=True):
    """
    Utility function to pad truncated frequency domain of real functions with zeros to a specified resolution
    :param F: n_dims tuple of period in each dimension
    :param res: n_dims int tuple of number of desired frequency modes
    :param norm: normalize output to have same physical density as original signal

    :return: F_pad: padded frequencies
    """
    assert(len(res) <= len(F.shape))
    for rprev, rnew in zip(F.shape, res[:-1]):
        assert(rnew >= rprev)
        assert((rnew - rprev) % 2 == 0)
    assert(res[-1]/2 >= F.shape[len(res)-1])
    pad_list = []
    for i in range(len(res)-1):
        pad_list.append((int((res[i] - F.shape[i])/2), int((res[i] - F.shape[i])/2)))
    pad_list.append((0, int((res[i]-F.shape[i])/2)))
    for i in range(len(res), len(F.shape)):
        pad_list.append((0, 0))
    F = np.fft.fftshift(F, axes=tuple(range(len(res)-1)))
    F_pad = np.pad(F, tuple(pad_list), 'constant')
    F_pad = np.fft.ifftshift(F_pad, axes=tuple(range(len(res)-1)))

    if norm:
        scale = 1
        for i in range(len(res)-1):
            scale *= res[i] / F.shape[i]
        scale *= res[-1]/ 2 /F.shape[i]
        F_pad *= scale

    return F_pad


def fftfreqs(res):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :param t: n_dims tuple of period in each dimension
    :return:
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(freq)
    r_ = res[-1]
    freqs.append(np.fft.rfftfreq(r_, d=1/r_)[:-1])
    omega = np.meshgrid(*freqs, indexing='ij')
    omega = list(omega)
    omega = np.stack(omega, axis=-1)

    return omega


def v2e_list(V, E):
    """
    Helper function to return V to E correspondence list for efficient computing
    :param V: vertex matrix. float ndarray of shape (n_vertex, n_dims)
    :param E: element matrix. int ndarray of shape (n_edge, 2)
    :return: v2elist: that maps vertex id to elements containing it

    """
    v2elist = [[] for _ in range(V.shape[0])]

    for ie in range(E.shape[0]):
        for iv in E[ie]:
            v2elist[iv].append(ie)
    for iv in range(V.shape[0]):
        v2elist[iv] = list(np.unique(v2elist[iv]))

    return v2elist


def simplex_content(j, n_dims, signed, *args):
    """
    Compute simplex content (j-dim-volume) using Cayley-Menger Determinant
    :param j: dimension of simplex
    :param n_dims: dimension of R^n space
    :param signed: bool denoting whether to calculate signed content or unsigned content
    :param args: v0, v1, ... vectors for the coordinates of the vertices defining the simplex

    :return: vol: volume of the simplex
    """
    assert(n_dims == len(args[0]))
    assert(isinstance(signed, bool))
    n_vert = len(args)
    assert(n_vert == j+1)

    if n_dims > j:
        assert(not signed)

    if not signed:
        # construct Cayley-Menger matrix
        B = np.zeros([j+2, j+2])
        B[:, 0] = 1
        B[0, :] = 1
        B[0, 0] = 0
        for r in range(1, j+2):
            for c in range(r+1, j+2):
                vr = args[r-1]
                vc = args[c-1]
                B[r, c] = np.linalg.norm(vr-vc) ** 2
                B[c, r] = B[r, c]
        vol2 = (-1)**(j+1) / (2**j) / (factorial(j)**2) * np.linalg.det(B)
        if vol2 < 0:
            print("Warning: zeroing small negative number {0}".format(vol2))
        vol = np.sqrt(max(0, vol2))
    else:
        # matrix determinant
        mat = np.zeros([j, j])
        for r in range(j):
            mat[:, r] = args[r] - args[-1]
        vol = np.linalg.det(mat) / factorial(j)

    return vol


def simplex_integral(j, *args):
    """
    Fourier integral of each simplex given list of sig and esig
    :param j: dimension of simplex
    :param args: (sig1, esig1), (sig2, esig2), ...

    :return: Fi: frequencies corresponding to each simplex
    """
    Fi = np.zeros(args[0][0].shape, dtype=np.complex_)
    if j > 0:
        for dim in range(j+1):
            other_dims = [d for d in range(j+1) if d != dim]
            denom = 1
            for d in other_dims:
                denom *= args[dim][0] - args[d][0]
            Fi += args[dim][1] / denom
        Fi[tuple([0] * len(Fi.shape))] = - 1 / factorial(j)
    else:
        Fi = args[0][1]
    Fi *= 1j**j
    return Fi


def simplex_ft_cpu(V, E, D, res, t, j, mode='density', prog_bar=False):
    """
    Fourier transform for signal defined on a j-simplex set in R^n space
    :param V: vertex list. float ndarray of shape (n_vertex, n_dims)
    :param E: element list. int ndarray of shape (n_edge, j or j+1)
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
    n_dims = V.shape[1]
    assert(n_dims == len(res))  # consistent spacial dimensionality
    assert(E.shape[0] == D.shape[0])  # consistent vertex numbers
    assert(mode in ['density', 'mass'])

    # number of columns in E
    ghost = E.shape[1] == j and n_dims == j
    assert (E.shape[1] == j+1 or ghost)
    if ghost:
        V = np.append(V, np.zeros([1, n_dims]), axis=0)
        E = np.append(E, V.shape[0] - 1 + np.zeros([E.shape[0], 1], dtype=np.int), axis=1)
    n_elem = E.shape[0]
    n_vert = V.shape[0]
    n_channel = D.shape[1]

    # frequency tensor
    omega = fftfreqs(res)
    omega[tuple([0] * n_dims)] += 1e-3 # will get rid of this

    # normalize frequencies
    for dim in range(n_dims):
        omega[..., dim] *= 2 * np.pi / t[dim]

    # initialize output F
    F_shape = list(omega.shape)[:-1]
    F_shape.append(n_channel)
    F = np.zeros(F_shape, dtype=np.complex_)

    # create v2e list
    v2elist = v2e_list(V, E)
    node_queue = deque([])
    sig_list = [None] * n_vert
    esig_list = [None] * n_vert
    vert_in_queue = [False] * n_vert
    elem_count = 0

    # progress bar
    if prog_bar:
        pbar = tqdm(total=n_elem)

    # BFS - breadth first search
    while elem_count < n_elem:
        if len(node_queue) == 0:
            for iv, vlist in enumerate(v2elist):
                if len(vlist) > 0:
                    node_queue.append(iv)
                    vert_in_queue[iv] = True
                    break
        iv = node_queue.popleft()
        if len(v2elist[iv]) != 0:
            elems = deepcopy(v2elist[iv])
            for ie in elems:
                verts = E[ie]
                for vert in verts:
                    v2elist[vert].remove(ie)
                    if not vert_in_queue[vert]:
                        node_queue.append(vert)
                        vert_in_queue[vert] = True
                    if sig_list[vert] is None:
                        sig_list[vert] = np.sum(V[vert] * omega, axis=-1)
                        esig_list[vert] = np.exp(-1j * sig_list[vert])
                vert_arrays = tuple([V[vert] for vert in verts])
                detJ = factorial(j) * simplex_content(j, n_dims, ghost, *vert_arrays)
                sig_tuples = tuple([(sig_list[vert], esig_list[vert]) for vert in verts])
                F0 = detJ * simplex_integral(j, *sig_tuples)
                for ic in range(n_channel):
                    F[..., ic] += F0 * D[ie, ic]
                elem_count += 1
                # progress bar
                if prog_bar:
                    pbar.update(1)
        # clear sig_list for this vertex
        sig_list[iv] = None
        esig_list[iv] = None
        gc.collect()

    # progress bar
    if prog_bar:
        pbar.close()

    if mode == 'density':
        if not np.array_equal(res, res[0]*np.ones(len(res))):
            print("WARNING: density preserving mode not correctly implemented if not all res are equal")
        F *= res[0] ** j
    return F


######################### GPU (CUDA) accelerated forward implementation ############################
BDIM2 = (16, 8)
GDIM2 = (32, 16)
BDIM3 = (8, 8, 4)
GDIM3 = (16, 16, 8)


def simplex_ft_gpu(V, E, D, res, t, j, mode='density', gpuid=0):
    """
    Fourier transform for signal defined on a j-simplex set in R^n space
    :param V: vertex list. float ndarray of shape (n_vertex, n_dims)
    :param E: element list. int ndarray of shape (n_edge, j or j+1)
              if j cols, then assume sign of area using RHR and use ghost node.
    :param D: int ndarray of shape (n_vertex, n_channel)
    :param res: n_dims int tuple of number of frequency modes
    :param t: n_dims tuple of period in each dimension
    :param j: dimension of simplex set
    :param mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
    :param gpuid: gpu device id

    :return: F: ndarray of shape (res[0], res[1], ..., res[-1]/2+1, n_channel)
                last dimension is halfed since the signal is assumed to be real
     """
    cuda.select_device(gpuid)
    n_dims = V.shape[1]
    assert(n_dims in [2, 3])  # GPU implementation not yet implemented for other dimensions
    assert(n_dims == len(res))  # consistent spacial dimensionality
    assert(E.shape[0] == D.shape[0])  # consistent vertex numbers
    assert(mode in ['density', 'mass'])

    # number of columns in E
    ghost = E.shape[1] == j and n_dims == j
    assert (E.shape[1] == j+1 or ghost)
    if ghost:
        V = np.append(V, np.zeros([1, n_dims]), axis=0)
        E = np.append(E, V.shape[0] - 1 + np.zeros([E.shape[0], 1], dtype=np.int), axis=1)
    n_elem = E.shape[0]
    n_vert = V.shape[0]
    n_channel = D.shape[1]

    # normalize frequencies
    omega = np.array([2*np.pi/ti for ti in t])

    # initialize output F
    F_shape = list(res)
    F_shape[-1] = int(res[-1]/2)
    F_shape.append(n_channel)
    F = np.zeros(F_shape, dtype=np.complex128)

    # compute content array and P array
    C = np.zeros(n_elem)
    P = np.zeros([n_elem, j+1, n_dims])

    for ie in range(n_elem):
        # content array
        verts = E[ie]
        vert_arrays = tuple([V[vert] for vert in verts])
        C[ie] = factorial(j) * simplex_content(j, n_dims, ghost, *vert_arrays)
        # p array
        P[ie] = V[E[ie]]

    # move arrays to device
    V_mem = cuda.to_device(V)
    E_mem = cuda.to_device(E)
    D_mem = cuda.to_device(D)
    C_mem = cuda.to_device(C)
    P_mem = cuda.to_device(P)
    F_mem = cuda.to_device(F)
    omega_mem = cuda.to_device(omega)

    # invoke kernel function
    if n_dims == 2:
        simplex_ft_kernel2[GDIM2, BDIM2](V_mem, E_mem, D_mem, C_mem, P_mem, omega_mem, F_mem)
    elif n_dims == 3:
        simplex_ft_kernel3[GDIM3, BDIM3](V_mem, E_mem, D_mem, C_mem, P_mem, omega_mem, F_mem)
    F_mem.to_host()
    F *= 1j**j

    # ifftshift
    F = np.fft.ifftshift(F, axes=tuple(range(n_dims-1)))

    if mode == 'density':
        if not np.array_equal(res, res[0]*np.ones(len(res))):
            print("WARNING: density preserving mode not correctly implemented if not all res are equal")
        F *= res[0] ** j
    return F

@cuda.jit
def simplex_ft_kernel2(V, E, D, C, P, omega, F):
    r = F.shape[:-1]
    j = E.shape[1] - 1
    n_elem = E.shape[0]
    n_dims = V.shape[1]
    n_channel = D.shape[1]

    # position within a grid
    start_u, start_v = cuda.grid(2)

    # u, v, w strides
    stride_u = cuda.gridDim.x * cuda.blockDim.x
    stride_v = cuda.gridDim.y * cuda.blockDim.y

    uv = cuda.local.array(shape=(2), dtype=nb.float64)

    for i in range(n_elem):
        for iu in range(start_u, int(r[0]), stride_u):
            u = (iu - r[0] / 2) * omega[0]
            for iv in range(start_v, int(r[1]), stride_v):
                v = iv * omega[1]
                uv[0] = u
                uv[1] = v
                for ic in range(n_channel):
                    if not ( u == 0 and v == 0):
                        F[iu, iv, ic] += simplex_ft_device(P[i], uv) * C[i] * D[i, ic]
                    else:
                        F[iu, iv, ic] += - 1 / 2 * C[i] * D[i, ic]
                cuda.syncthreads()

@cuda.jit
def simplex_ft_kernel3(V, E, D, C, P, omega, F):
    r = F.shape[:-1]
    j = E.shape[1] - 1
    n_elem = E.shape[0]
    n_dims = V.shape[1]
    n_channel = D.shape[1]

    # position within a grid
    start_u, start_v, start_w = cuda.grid(3)

    # u, v, w strides
    stride_u = cuda.gridDim.x * cuda.blockDim.x
    stride_v = cuda.gridDim.y * cuda.blockDim.y
    stride_w = cuda.gridDim.z * cuda.blockDim.z

    uvw = cuda.local.array(shape=(3), dtype=nb.float64)

    for i in range(n_elem):
        for iu in range(start_u, int(r[0]), stride_u):
            u = (iu - r[0] / 2) * omega[0]
            for iv in range(start_v, int(r[1]), stride_v):
                v = (iv - r[1] / 2) * omega[1]
                for iw in range(start_w, int(r[2]), stride_w):
                    w = iw * omega[2]
                    uvw[0] = v
                    uvw[1] = u
                    uvw[2] = w
                    for ic in range(n_channel):
                        if not ( u == 0 and v == 0 and w == 0):
                            F[iu, iv, iw, ic] += simplex_ft_device(P[i], uvw) * C[i] * D[i, ic]
                        else:
                            F[iu, iv, iw, ic] += - 1 / 6 * C[i] * D[i, ic]
                    cuda.syncthreads()

@cuda.jit(device=True)
def simplex_ft_device(p, uvw):
    j = p.shape[0]-1
    n_dims = len(uvw)
    Fi = complex(0, 0)
    for dim in range(j+1):
        xi = p[dim]
        denom = 1
        uvw_dot_xi = complex(0, 0)
        for dd in range(n_dims):
            uvw_dot_xi += uvw[dd] * xi[dd]
        for d in range(j+1):
            if d != dim:
                xj = p[d]
                uvw_dot_xj = complex(0, 0)
                for dd in range(n_dims):
                    uvw_dot_xj += uvw[dd] * xj[dd]
                denom *= uvw_dot_xi - uvw_dot_xj
        Fi += exp(-1j*(uvw_dot_xi)) / denom
    return Fi

########################## GPU (CUDA) accelerated backward method implementation #########################
BDIM2 = (16, 8)
GDIM2 = (32, 16)
BDIM3 = (8, 8, 4)
GDIM3 = (16, 16, 8)


def simplex_ft_bw_gpu(dF, V, E, D, res, t, j, mode='density', gpuid=0):
    """
    Fourier transform backwards method for signal defined on a j-simplex set in R^n space
    :param dF: sensitivities on frequency modes (res[0], res[1], ..., res[-1]/2+1, n_channel)
    :param V: vertex list. float ndarray of shape (n_vertex, n_dims)
    :param E: element list. int ndarray of shape (n_edge, j or j+1)
              if j cols, then assume sign of area using RHR and use ghost node.
    :param D: int ndarray of shape (n_vertex, n_channel)
    :param res: n_dims int tuple of number of frequency modes
    :param t: n_dims tuple of period in each dimension
    :param j: dimension of simplex set
    :param mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
    :param gpuid: gpu device id

    :return: dV: output tensor of shape (n_vertex, n_dims)
     """
    cuda.select_device(gpuid)
    n_dims = V.shape[1]
    assert(n_dims in [2])  # GPU implementation not yet implemented for other dimensions
    assert(j == 2) # not yet implemented for other simplices
    assert(n_dims == len(res))  # consistent spacial dimensionality
    assert(E.shape[0] == D.shape[0])  # consistent vertex numbers
    assert(mode in ['density', 'mass'])

    # number of columns in E
    ghost = E.shape[1] == j and n_dims == j
    assert (E.shape[1] == j+1 or ghost)
    if ghost:
        V = np.append(V, np.zeros([1, n_dims]), axis=0)
        E = np.append(E, V.shape[0] - 1 + np.zeros([E.shape[0], 1], dtype=np.int), axis=1)
    n_elem = E.shape[0]
    n_vert = V.shape[0]
    n_channel = D.shape[1]

    # normalize frequencies
    omega = np.array([2*np.pi/ti for ti in t])

    # initialize output dV
    dV_shape = list(V.shape)
    dFdV_shape = tuple([j+1, n_dims] + list(dF.shape))
    dV = np.zeros(dV_shape, dtype=np.complex128)
    dFdV = np.zeros(dFdV_shape, dtype=np.complex128)

    # compute content array and P array
    C = np.zeros(n_elem)
    P = np.zeros([n_elem, j+1, n_dims])

    for ie in range(n_elem):
        # content array
        verts = E[ie]
        vert_arrays = tuple([V[vert] for vert in verts])
        C[ie] = factorial(j) * simplex_content(j, n_dims, ghost, *vert_arrays)
        # p array
        P[ie] = V[E[ie]]

    # fftshift for dF
    dF = np.fft.fftshift(dF, axes=tuple(range(n_dims-1)))
        
    # move arrays to device
    V_mem = cuda.to_device(V)
    E_mem = cuda.to_device(E)
    D_mem = cuda.to_device(D)
    C_mem = cuda.to_device(C)
    P_mem = cuda.to_device(P)
    dF_mem = cuda.to_device(dF)
    dFdV_mem = cuda.to_device(dFdV)
    omega_mem = cuda.to_device(omega)

    sum_reduce = cuda.reduce(lambda a, b: a + b)
    # invoke kernel function
    if n_dims == 2:
        for ie in range(n_elem):
            simplex_ft_bw_kernel2[GDIM2, BDIM2](ie, V_mem, E_mem, D_mem, C_mem, P_mem, omega_mem, dF_mem, dFdV_mem)
            for ip in range(j+1):
                for idim in range(n_dims):
                    tmp_sum = sum_reduce(dFdV_mem[ip, idim].ravel())
                    dV[E[ie, ip], idim] += tmp_sum
        
    elif n_dims == 3:
        print("TODO: Not implemented for 3D yet...")

    if mode == 'density':
        if not np.array_equal(res, res[0]*np.ones(len(res))):
            print("WARNING: density preserving mode not correctly implemented if not all res are equal")
        dV *= res[0] ** j
    return dV

@cuda.jit
def simplex_ft_bw_kernel2(ie, V, E, D, C, P, omega, dF, dFdV):
    r = dF.shape[:-1]
    j = 2
    n_elem = E.shape[0]
    n_dims = V.shape[1]
    n_channel = D.shape[1]

    # position within a grid
    start_u, start_v = cuda.grid(2)

    # u, v strides
    stride_u = cuda.gridDim.x * cuda.blockDim.x
    stride_v = cuda.gridDim.y * cuda.blockDim.y

    uv = cuda.local.array(shape=(2), dtype=nb.float64)

    for iu in range(start_u, int(r[0]), stride_u):
        u = (iu - r[0] / 2) * omega[0]
        for iv in range(start_v, int(r[1]), stride_v):
            v = iv * omega[1]
            uv[0] = u
            uv[1] = v
            for ic in range(n_channel):
                fac = simplex_ft_bw_device2(P[ie], uv, dF[iu, iv, ic], C[ie])
                ct = 0
                for ip in range(j+1):
                    dFdV[ip, 0, iu, iv, ic] = fac[ct]
                    dFdV[ip, 1, iu, iv, ic] = fac[ct+1]
                    ct += 2


@cuda.jit(device=True)
def simplex_ft_bw_device2(p, uv, df, c):
    j = p.shape[0]-1
    n_dims = len(uv)
    if not (uv[0] == 0 and uv[1] == 0):
        sig0 = 0
        sig1 = 0
        sig2 = 0
        for d in range(n_dims):
            sig0 += p[0, d] * uv[d]
            sig1 += p[1, d] * uv[d]
            sig2 += p[2, d] * uv[d]
        s01 = 1/(sig0-sig1)
        s12 = 1/(sig1-sig2)
        s20 = 1/(sig2-sig0)
        e0 = -exp(-1j*sig0)*s01*s20
        e1 = -exp(-1j*sig1)*s12*s01
        e2 = -exp(-1j*sig2)*s20*s12
        fac0 = e0*(-1j-s01+s20)-e1*s01+e2*s20
        fac1 = e1*(-1j-s12+s01)-e2*s12+e0*s01
        fac2 = e2*(-1j-s20+s12)-e0*s20+e1*s12
        S = e0+e1+e2
    else: # take the limit of uv->0
        fac0 = 0
        fac1 = 0
        fac2 = 0
        S = -0.5
        
    fac = (fac0, fac1, fac2)
    # sorry for the ugly code below. numba doesn't allow lists inside device and it's a real pain.
    ip = 0
    pnx  = p[(ip+1)%(j+1), 0]
    pny  = p[(ip+1)%(j+1), 1]
    pnnx = p[(ip+2)%(j+1), 0]
    pnny = p[(ip+2)%(j+1), 1]
    res00 = df*(-uv[0]*fac[ip]*c-S*(pny-pnny))
    res01 = df*(-uv[1]*fac[ip]*c-S*(-pnx+pnnx))
    ip = 1
    pnx  = p[(ip+1)%(j+1), 0]
    pny  = p[(ip+1)%(j+1), 1]
    pnnx = p[(ip+2)%(j+1), 0]
    pnny = p[(ip+2)%(j+1), 1]
    res10 = df*(-uv[0]*fac[ip]*c-S*(pny-pnny))
    res11 = df*(-uv[1]*fac[ip]*c-S*(-pnx+pnnx))
    ip = 2
    pnx  = p[(ip+1)%(j+1), 0]
    pny  = p[(ip+1)%(j+1), 1]
    pnnx = p[(ip+2)%(j+1), 0]
    pnny = p[(ip+2)%(j+1), 1]
    res20 = df*(-uv[0]*fac[ip]*c-S*(pny-pnny))
    res21 = df*(-uv[1]*fac[ip]*c-S*(-pnx+pnnx))
    return res00, res01, res10, res11, res20, res21
