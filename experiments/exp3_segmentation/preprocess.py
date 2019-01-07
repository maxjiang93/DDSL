import numpy as np 
import argparse
from tqdm import tqdm
import torch
import sys; sys.path.append("../../ddsl")
from ddsl import DDSL_phys
import os
from glob import glob

# Class label mapping
mapping =  {"car": 0,
            "truck": 1,
            "train": 2,
            "bus": 3,
            "motorcycle": 4,
            "bicycle": 5,
            "rider": 6,
            "person": 7}

EPS = 1e-4

def convert_ddsl(V, res=224):
    V = torch.DoubleTensor(V)
    npoints = V.shape[0]
    perm = list(range(1, npoints)) + [0]
    seq0 = torch.arange(npoints)
    seq1 = seq0[perm]
    E = torch.stack((seq0, seq1), dim=-1)
    D = torch.ones(E.shape[0], 1, dtype=torch.float64)
    ddsl = DDSL_phys([res] * 2, [1] * 2, j=2)
    V += 1e-4 * torch.rand_like(V)
    V, E, D = V.cuda(), E.cuda(), D.cuda()
    f = ddsl(V, E, D)
    return f.squeeze().detach().cpu().numpy()

def process_one(infile, outfile, res=224):
    dat = np.load(infile)
    V = dat['poly']
    # check poly within [0,1)
    assert(V.min() >= 0 and V.max() < 1)
    f = convert_ddsl(V, res=res)
    # check orientation
    if np.absolute(f.max()) < np.absolute(f.min()):
        f = -f
    f = f.T
    save_dict = {'image': dat['image'],
                 'label': dat['label'],
                 'label_id': mapping[str(dat['label'])],
                 'target' : f,
                 'img_path': dat['img_path']}
    np.savez_compressed(outfile, **save_dict)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--data_dir', type=str, default="../../data/original_data", metavar='I', help="original data directory")
    parser.add_argument('--out_dir', type=str, default="../../data/processed_data", metavar='O', help="output directory")
    parser.add_argument('--res', type=int, default=224, metavar='R', help="raster resolution")
    parser.add_argument('--no_prog_bar', action='store_true', help="disable progress bar.")

    args = parser.parse_args()

    partitions = ['test', 'train', 'val']

    # create output directory
    for p in partitions:
        odir = os.path.join(args.out_dir, p)
        if not os.path.exists(odir):
            os.makedirs(odir)

    od_ = args.out_dir.split('/')[-1]
    dd_ = args.data_dir.split('/')[-1]
    infiles = glob(os.path.join(args.data_dir, "*/*.npz"))
    outfiles = [f.replace(dd_, od_) for f in infiles]

    if not args.no_prog_bar:
        pbar = tqdm(total=len(infiles))

    for ifile, ofile in zip(infiles, outfiles):
        if not os.path.exists(ofile):
            try:
                process_one(ifile, ofile, res=args.res)
            except:
                print("Failed to process {}".format(ifile))
        if not args.no_prog_bar:
            pbar.update(1)

    if not args.no_prog_bar:
        pbar.close()

if __name__ == '__main__':
    main()

   