import numpy as np
from shapely import geometry, affinity, wkt
from sklearn.model_selection import train_test_split
import json
import pickle
# import sys; sys.path.append("../../methods")
import sys; sys.path.append("../../ddsl")
# from transform import simplex_ft_cpu
from ddsl import *
from torch.utils.data import Dataset
import os
from shutil import rmtree


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

# data loader module
class PMNISTDataSet(Dataset):
    def __init__(self, path, partition='train', imsize=28, cache_root='./cache'):
        """
        Args:
            path (string): Directory containing polygon and label json files.
            partition (string): train or test
            imsize (int): imsize * imsize is the scale to feed output images
        """
        assert(partition in ['train', 'test'])
        
        self.partition = partition
        self.imsize = imsize
        self.cache_root = cache_root
        self.cache_dir = os.path.join(self.cache_root, partition+'_'+str(imsize))
        
        # load labels and input polygons
        if partition == 'test':
            with open(os.path.join(path, "mnist_polygon_test.json"), 'r') as infile:
                self.plist = json.load(infile)
            with open(os.path.join(path, "mnist_label_test.json"), 'r') as infile:
                self.label = json.load(infile)
        else:
            with open(os.path.join(path, "mnist_polygon_train.json"), 'r') as infile:
                self.plist = json.load(infile)
            with open(os.path.join(path, "mnist_label_train.json"), 'r') as infile:
                self.label = json.load(infile)

        # cache options
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def __len__(self):
        return len(self.plist)

    def __getitem__(self, idx):
        cache_file = os.path.join(self.cache_dir, 'cache_{:05d}.pkl'.format(idx))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                try:
                    r = pickle.load(f)
                except EOFError:
                    print("EOF ERROR while accessing {0}".format(cache_file))
                    r = self._cache(idx)
            return r
        else:
            r = self._cache(idx)
            return r
    
    def _cache(self, idx):
        cache_file = os.path.join(self.cache_dir, 'cache_{:05d}.pkl'.format(idx))
        P = wkt.loads(self.plist[idx])
        V, E = poly2ve(P)
        V += 1e-6*np.random.rand(*V.shape)
        V=torch.tensor(V, dtype=torch.float64, requires_grad=False)
        E=torch.LongTensor(E)
        D = torch.ones(E.shape[0], 1, dtype=torch.float64)
        ddsl_phys=DDSL_phys((self.imsize,self.imsize),(1,1),2,1)
        input_ = ddsl_phys(V,E,D)
        label_ = self.label[idx]
        r = {"input": input_, 
             "label": label_}
        with open(cache_file, 'wb') as f:
            pickle.dump(r, f)
        return r
    
    def _clean_cache(self):
        rmtree(self.cache_dir)
