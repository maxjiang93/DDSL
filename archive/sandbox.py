def create_shape(dim, j):
    assert(j in [0, 1, 2, 3])
    assert(dim in [2, 3])
    assert(j <= dim)
    if dim == 3:
        V = [[1, 1, 1],
             [1, -1, -1],
             [-1, 1, -1],
             [-1, -1, 1]]
        if j == 0:
            E = [[0],
                 [1],
                 [2],
                 [3]]
        elif j == 1:
            E = [[0,1],
                 [0,2],
                 [0,3],
                 [1,2],
                 [1,3],
                 [2,3]]
        elif j == 2:
            E = [[0,1,2],
                 [0,1,3],
                 [0,2,3],
                 [1,2,3]]
        elif j == 3:
            E = [[0,1,2,3]]
    elif dim == 2:
        V = [[.25, .25],
             [.75, .25],
             [.50, .75]]
        if j == 0:
            E = [[0],
                 [1],
                 [2]]
        elif j == 1:
            E = [[0,1],
                 [0,2],
                 [1,2]]
        elif j == 2:
            E = [[0,1,2]]
    V = torch.FloatTensor(V)
    E = torch.LongTensor(E)
    
    # normalize V
    V_bb = torch.max(V, dim=0)[0] - torch.min(V, dim=0)[0]
    V_c = (torch.max(V, dim=0)[0] + torch.min(V, dim=0)[0]) / 2
    V -= V_c
    V /= 1.5*V_bb
    V += 0.5
    return V, E


def simplex_content_old(V, E):
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
        