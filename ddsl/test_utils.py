import numpy as np
import torch
from scipy.spatial import Delaunay
from time import time
from math import ceil

from ddsl import DDSL_spec

def rand_hull(n_points, dim, hull=False, dtype=torch.float32):
    V = np.random.rand(n_points, dim)
    mesh = Delaunay(V)
    E = mesh.simplices
    V, E = torch.tensor(V, dtype=dtype), torch.LongTensor(E)
    # normalize V to 
    V_bb = torch.max(V, dim=0)[0] - torch.min(V, dim=0)[0]
    V_c = (torch.max(V, dim=0)[0] + torch.min(V, dim=0)[0]) / 2
    V -= V_c
    V /= 1.5*V_bb
    V += 0.5
    return V, E
    
def generate_random_mesh(j, dim, npoints, n_chan=1, uniform_density=False, device="cuda"):
    assert(j <= dim)
    dev = torch.device(device)
    
    if j == 0: # points
        V = torch.rand(npoints, dim, dtype=torch.float64)
        E = torch.arange(V.shape[0]).unsqueeze(-1)
        V, E = V.cuda(), E.cuda()
        V.requires_grad = True
    elif j == 1: # lines
        npoints = ceil(npoints / 2) * 2
        V = torch.rand(npoints, dim, dtype=torch.float64)
        E = torch.arange(V.shape[0]).view(-1, 2)
    elif j == 2: # triangles
        if dim == 2:
            V, E = rand_hull(npoints, dim, dtype=torch.float64)
        else:
            npoints = ceil(npoints / 3) * 3
            V = torch.rand(npoints, dim, dtype=torch.float64)
            E = torch.arange(V.shape[0]).view(-1, 3)
    elif j == 3: # tetrahedron
        if dim == 3:
            V, E = rand_hull(npoints, dim, dtype=torch.float64)
        else:
            npoints = ceil(npoints / 4) * 4
            V = torch.rand(npoints, dim, dtype=torch.float64)
            E = torch.arange(V.shape[0]).view(-1, 4)
            
    if uniform_density:
        D = torch.ones(E.shape[0], n_chan, dtype=V.dtype)
    else:
        D = torch.rand(E.shape[0], n_chan, dtype=V.dtype)
    V, E, D = V.to(dev), E.to(dev), D.to(dev)
    V.requires_grad = True
    D.requires_grad = True
    
    return V, E, D

def test_ddsl(j, dim, npoints, res, n_chan=1, accu=0.1, print_stats=False):
    # generate a random mesh
    V, E, D = generate_random_mesh(j, dim, npoints, n_chan=n_chan)

    res = [res] * dim
    t = [1] * dim

    # Sensitivity on each frequency mode
    r = res[0]
    if dim == 2:
        dF = torch.rand(r, int(r/2)+1, n_chan, 2, dtype=V.dtype, device=V.device) # unit sensitivity
    elif dim == 3:
        dF = torch.rand(r, r, int(r/2)+1, n_chan, 2, dtype=V.dtype, device=V.device) # unit sensitivity
    else:
        print("Test not implemented for dimensions other than 2 and 3.")

    # Analytical Adjoint solution
    ddsl = DDSL_spec(res,t,j,1,'density')
    t0 = time()
    F = ddsl(V,E,D)
    t1 = time()
    loss = (F*dF).view(-1).sum(0)
    if V.grad is not None:
        V.grad.zero_()
    t2 = time()
    loss.backward()
    t3 = time()
    dV = V.grad
    dD = D.grad

    # Finite difference approximation
    t4 = time()
    delta = 1e-6
    dV_fd_all = torch.zeros(*(list(V.shape)+list(dF.shape)), dtype=V.dtype).cuda()
    dD_fd_all = torch.zeros(*(list(D.shape)+list(dF.shape)), dtype=V.dtype).cuda()
    for ii in range(V.shape[0]):
        for jj in range(V.shape[1]):
            V_p = V.clone()
            V_m = V.clone()
            V_p[ii, jj] += delta
            V_m[ii, jj] -= delta
            Freq_p = ddsl(V_p, E, D)
            Freq_m = ddsl(V_m, E, D)
            dV_fd_all[ii,jj] = (Freq_p - Freq_m) / delta / 2
    for ii in range(D.shape[0]):
        for jj in range(D.shape[1]):
            D_p = D.clone()
            D_m = D.clone()
            D_p[ii, jj] += delta
            D_m[ii, jj] -= delta
            Freq_p = ddsl(V, E, D_p)
            Freq_m = ddsl(V, E, D_m)
            dD_fd_all[ii,jj] = (Freq_p - Freq_m) / delta / 2
            
    dV_fd = (dV_fd_all * dF).view(V.shape[0], V.shape[1], -1).sum(-1)
    dD_fd = (dD_fd_all * dF).view(D.shape[0], D.shape[1], -1).sum(-1)
    t5 = time()

    if print_stats:
        # Print analysis
        print("/ ***** Runtime Analysis ***** /")
        print("Resolution: {}, # Elements: {}, Simplex Degree j: {}".format(res, E.shape[0], E.shape[1]-1))
        print("Forward     Time: {}".format(t1-t0))
        print("Analytical  Backward Time: {}".format(t3-t2))
        print("Finite-Diff Backward Time: {}".format(t5-t4))

        print("\n/ ***** Correctness ***** /")
        print("Analytical Gradient Matches Finite Difference Gradient to Accuracy {}:".format(accu))
        print("V-gradient:")
        print((torch.abs(dV - dV_fd) < accu).detach().cpu().numpy() == 1)
        print("D-gradient:")
        print((torch.abs(dD - dD_fd) < accu).detach().cpu().numpy() == 1)
        print("\n Analytical Gradient:")
        print("V-gradient:")
        print(dV.detach().cpu().numpy())
        print("D-gradient:")
        print(dD.detach().cpu().numpy())
        print("Finite Difference Gradient:")
        print("V-gradient:")
        print(dV_fd.detach().cpu().numpy())
        print("D-gradient:")
        print(dD_fd.detach().cpu().numpy())
    
    match_V = torch.abs(dV - dV_fd) < accu
    match_D = torch.abs(dD - dD_fd) < accu
    pass_test_dV = (torch.sum(match_V) == match_V.numel()).detach().cpu().item() == 1
    pass_test_dD = (torch.sum(match_D) == match_D.numel()).detach().cpu().item() == 1
    
    return pass_test_dV, pass_test_dD