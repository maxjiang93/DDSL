from shapely import geometry, affinity, wkt
from scipy.misc import imsave
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mnist import MNIST
from tqdm import tqdm
import os
import json
from joblib import Parallel, delayed
from math import ceil
import os

def poly2ve(Poly):
    '''Function to convert polygon or multipolygon types in shapely to vertex and edge matrices'''
    
    def edgelist(startid, length, flip=False):
        # helper function to create edge list
        p1 = np.arange(startid, startid+length)
        p2 = p1 + 1
        p2[-1] = startid
        if not flip:
            return np.stack((p1, p2), axis=-1)
        else:
            return np.flipud(np.stack((p2, p1), axis=-1))
    
    def singlePolygon(P):
        # helper function for processing a single polygon instance
        assert(isinstance(P, geometry.polygon.Polygon))
        v = []
        e = []
        ecount = 0
        # exterior
        v_ex = np.array(P.exterior)[:-1]
        e_ex = edgelist(0, v_ex.shape[0])
        v.append(v_ex)
        e.append(e_ex)
        ecount += v_ex.shape[0]
        # interiors
        for int_ in P.interiors:
            v_in = np.array(int_)
            e_in = edgelist(ecount, v_in.shape[0], flip=False)
            v.append(v_in)
            e.append(e_in)
            ecount += v_in.shape[0]
        v = np.concatenate(v, axis=0)
        e = np.concatenate(e, axis=0)
        if not P.exterior.is_ccw:
            e = np.concatenate([e[:, 1:2], e[:, 0:1]], axis=-1) # flip e
        return v, e
         
    if isinstance(Poly, geometry.polygon.Polygon):
        V, E = singlePolygon(Poly)
    elif isinstance(Poly, geometry.multipolygon.MultiPolygon):
        V = []
        E = []
        ecount = 0
        for P in Poly.geoms:
            v, e = singlePolygon(P)
            V.append(v)
            E.append(e+ecount)
            ecount += v.shape[0]
        V = np.concatenate(V, axis=0)
        E = np.concatenate(E, axis=0)
    return V, E

def polyfft2(V, E, r=(128, 128), T=128):
    """
    V(float), E(int) are the vertex and edge matrices
    r is resolution len = 2 array
    Counter-clockwise is positive
    """
    assert V.shape[1] == 2
    if V.max()<=1:
        V *= T
    eps = 1e-5
    omega = 2 * np.pi / T
    V += eps * np.random.rand(*(V.shape))
    ulin = np.linspace(-r[0]/2, r[0]/2-1, r[0])
    vlin = np.linspace(0, r[1]/2, int(r[1]/2)+1)
    U_, V_ = np.meshgrid(ulin, vlin, indexing='ij')
    U_ += eps
    V_ += eps
    U_ *= omega
    V_ *= omega
    F = np.zeros(shape=U_.shape, dtype=np.complex_)
    for ei in range(E.shape[0]):
        x1, y1 = V[E[ei, 0], 0], V[E[ei, 0], 1]
        x2, y2 = V[E[ei, 1], 0], V[E[ei, 1], 1]
        Fn = np.exp(-1j*(U_*(x1+x2)+V_*(y1+y2)))*(np.exp(1j*(U_*x1+V_*y1))-np.exp(1j*(U_*x2+V_*y2)))*(x1-x2)/ \
             (V_*(U_*(x1-x2)+V_*(y1-y2)))
        F += Fn
    return F

def mnist2poly(image, hd_dim=64, wkt=False):
    image = np.flipud(image)
    # upsample in freq domain
    image_F = np.fft.fftshift(np.fft.rfft2(image), axes=(0))
    pad_w, pad_h = int((hd_dim-image_F.shape[0])/2), int(hd_dim/2+1-image_F.shape[1])
    image_F = np.pad(image_F, ((pad_w, pad_w), (0, pad_h)), 'constant')
    image_F = np.fft.ifftshift(image_F, axes=(0))
    image = np.fft.irfft2(image_F)
    cs = plt.contour(image, [(image.max()+image.min())/2])
    plt.close()
    polys = []
    
    for col in cs.collections:
        # Loop through all polygons that have the same intensity level
        for contour_path in col.get_paths(): 
            # Create the polygon for this intensity level
            # The first polygon in the path is the main one, the following ones are "holes"
            for ncp,cp in enumerate(contour_path.to_polygons()):
                new_shape = geometry.Polygon(cp)
                if ncp == 0:
                    poly = new_shape
                else:
                    # Remove the holes if there are any
                    poly = poly.difference(new_shape)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                polys.append(poly)
    P = polys[0]
    if len(polys) > 1:
        for i in range(1, len(polys)):
            P = P.difference(polys[i]).union(polys[i].difference(P))
    P = affinity.scale(P, 1/hd_dim, 1/hd_dim, 1/hd_dim, (0, 0, 0))
    if wkt:
        return P.wkt
    else:
        return P

def poly2bin(p, dim, output=None):
    N = dim
    xlin, ylin = np.linspace(0, 1, N), np.linspace(0, 1, N)
    X, Y = np.meshgrid(xlin, ylin)
    X, Y = X.ravel(), Y.ravel()
    Z = np.zeros(*X.shape).ravel()
    for i, (x, y) in enumerate(zip(X, Y)):
        Z[i] = float(P.contains(geometry.Point([x, y])))
    Z = Z.reshape([N, N])
    if output is None:
        return Z
    elif isinstance(output, str):
        imsave(output, Z)

def poly2phys(p, dim, upres=1, output=None):
    N = dim
    V, E = poly2ve(P)
    F = polyfft2(V, E, r=(N, N), T=N)
    if upres > 1:
        n = dim*upres
        w = int((n-dim)/2)
        F = np.pad(F, ((w, w), (0, w)), 'constant')
    F_ = np.fft.ifftshift(F, axes=(0))
    f = np.fft.irfft2(F_) * (upres) ** 2
    f = f.T
    if output is None:
        return f
    elif isinstance(output, str):
        imsave(output, f)

def process_mnist2poly(input_list, outfile, hd_dim=64):
    """
    Process images in mnist dataset given list of images
    Inputs:
    input_list: list containing images
    hd_dim: expansion dimension for extracting contours
    """
    P_list = [None] * len(input_list)
    pbar = tqdm(total=len(input_list))
    for i, im_list in enumerate(input_list):
        P_list[i] = mnist2poly(np.array(im_list).reshape([28, 28], order='C'), hd_dim, True)
        pbar.update(1)

    with open(outfile, 'w') as outputfile:
        json.dump(P_list, outputfile)
    if prog_bar:
        pbar.close()


def main():
    # load mnist
    data_dir = os.path.abspath(os.path.join(os.getcwd(), "data/MNIST"))
    mndata = MNIST(data_dir)
    mndata.gz = True
    images_train, labels_train = mndata.load_training()
    images_test, labels_test = mndata.load_testing()
    output_root = "data/polyMNIST"
    if not os.path.exists(output_root):
        os.mkdirs(output_root)
    train_file = os.path.join(output_root, "mnist_polygon_train.json")
    test_file = os.path.join(output_root, "mnist_polygon_test.json")
    print("Processing training files...")
    process_mnist2poly(images_train, outfile=train_file)
    print("Processing test files...")
    process_mnist2poly(images_test, outfile=test_file)
    # write labels to json file
    with open(os.path.join(output_root, "mnist_label_train.json"), 'w') as outputfile:
        json.dump(list(labels_train), outputfile)
    with open(os.path.join(output_root, "mnist_label_test.json"), 'w') as outputfile:
        json.dump(list(labels_test), outputfile)

if __name__ == '__main__':
    main()
