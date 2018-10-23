from glob import glob
import os
from tqdm import tqdm
import pickle
import numpy as np; np.set_printoptions(threshold=np.nan)
from sklearn.preprocessing import label_binarize


class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n

def compute_stats(data_dir):
    files = sorted(glob(os.path.join(data_dir, "*.pkl")))
    rolling_sum = 0
    sr = StatsRecorder()
    for f in tqdm(files):
        p = pickle.load(open(f, "rb"))['input']
        u = p[..., 0] + p[..., 1]*(1j)
        u = np.fft.irfft2(u, (28,28))
        u = np.expand_dims(u.ravel(), axis=-1)
        # from pdb import set_trace; set_trace()
        sr.update(u)
    print("Data Mean: \n", '[' + ', '.join(list(sr.mean.astype(str))) + ']') 
    print("Data std: \n", '[' + ', '.join(list(sr.std.astype(str))) + ']') 
    return sr.mean, sr.std

mean, std = compute_stats("cache/train_28")