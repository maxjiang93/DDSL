import sys
sys.path.append("../../ddsl")

import json
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import skimage
import argparse
import torch

from ddsl import DDSL_phys
import os
from glob import glob
import fileinput
import re

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

def convert_ddsl(V, res=224, dev=torch.device('cuda')):
    V = torch.DoubleTensor(V)
    npoints = V.shape[0]
    perm = list(range(1, npoints)) + [0]
    seq0 = torch.arange(npoints)
    seq1 = seq0[perm]
    E = torch.stack((seq0, seq1), dim=-1)
    D = torch.ones(E.shape[0], 1, dtype=torch.float64)
    ddsl = DDSL_phys([res] * 2, [1] * 2, j=2)
    V += 1e-4 * torch.rand_like(V)
    V, E, D = V.to(dev), E.to(dev), D.to(dev)
    f = ddsl(V, E, D)
    return f.squeeze().detach().cpu().numpy()

def replace( filePath, text, subs, flags=0 ):
    with open( filePath, "r+" ) as file:
        fileContents = file.read()
        textPattern = re.compile( re.escape( text ), flags )
        fileContents = textPattern.sub( subs, fileContents )
        file.seek( 0 )
        file.truncate()
        file.write( fileContents )

def main(args):
    # create output directory if it does not exist
    out_folder = args.out_folder
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available else "cpu")
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    opts = json.load(open('dataset.json', 'r'))['dataset']

    # overwrite dataset dir
    json_dir = os.path.join(args.polyrnn2_dir, 'data', 'cityscapes_final_v5')
    opts['train']['data_dir'] = json_dir
    opts['train_val']['data_dir'] = json_dir

    # run change path script to change paths for local computer
    script_path = os.path.join(args.polyrnn2_dir, 'code', 'Scripts', 'data', 'change_paths.py')
    city_dir = 'leftImg8bit'
    replace(script_path, "'rw'", "'r'")
    os.system("python {} --city_dir {} --json_dir {} --out_dir {}".format(script_path, city_dir, json_dir, json_dir))


    print("Building dataloaders...")

    dataset_train     = DataProvider(split='train', opts=opts['train'])
    dataset_train_val = DataProvider(split='train_val', opts=opts['train_val'])
    dataset_val       = DataProvider(split='val', opts=opts['train_val'])

    train_loader = DataLoader(dataset_train, batch_size=batch_size,
        shuffle = False, num_workers=num_workers, collate_fn=collate_fn)
    train_val_loader = DataLoader(dataset_train_val, batch_size=batch_size,
        shuffle = False, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=batch_size,
        shuffle = False, num_workers=num_workers, collate_fn=collate_fn)

    # progress bar
    ntotal = 40174+3448+8440
    pbar = tqdm(total=ntotal)
    res = 224

    loaders = {'train': train_loader,
               'val': train_val_loader,
               'test': val_loader}
    
    for key in loaders.keys():
        this_folder = os.path.join(out_folder, key)
        if not os.path.exists(this_folder):
            os.makedirs(this_folder)
        loader = loaders[key]
        count = 0
        for d in loader:
            for ind in range(batch_size):
                img = (d['img'][ind].numpy()*255).astype(np.uint8)
                label = d['label'][ind]
                poly = d['orig_poly'][ind]
                img_path = d['img_path'][ind].split('/data/')[-1]
  
                def adjust(f):
                    # check orientation
                    if np.absolute(f.max()) < np.absolute(f.min()):
                        f = -f
                    return f.T

                V = poly
                # check poly within [0,1)
                assert(V.min() >= 0 and V.max() < 1)
                f = adjust(convert_ddsl(V, res=res, dev=device))

                f_2 = adjust(convert_ddsl(V, res=int(res/2), dev=device))
                f_4 = adjust(convert_ddsl(V, res=int(res/4), dev=device))
                f_8 = adjust(convert_ddsl(V, res=int(res/8), dev=device))
                save_dict = {'image': img,
                             'label': label,
                             'label_id': mapping[str(label)],
                             'target' : f,
                             'target_2': f_2,
                             'target_4': f_4,
                             'target_8': f_8,
                             'img_path': img_path}
                
                fname = os.path.join(this_folder, '{0:05d}'.format(count)); count += 1
                np.savez_compressed(fname, **save_dict)
                pbar.update(1)
    pbar.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for exp3')
    parser.add_argument('--out_folder', type=str, default='mres_processed_data_test', metavar='O',
                        help='output folder name to store processed data')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size when preprocessing data')
    parser.add_argument('--num_workers', type=int, default=12, metavar='N',
                        help='number of cpu workers')
    parser.add_argument('--no_cuda', action='store_true', help='only use cpu')
    parser.add_argument('--polyrnn2_dir', type=str, default='polyrnn-pp-pytorch-small')
    parser.add_argument('--city_dir', type=str, default='leftImg8bit')
    
    args = parser.parse_args()
    
    # add data processing codes from polygonrnn++ directory
    assert(os.path.exists(args.city_dir) and os.path.exists(args.polyrnn2_dir))
    json_dir = os.path.join(args.polyrnn2_dir, 'data', 'cityscapes_final_v5')
    if not os.path.exists(json_dir):
        os.system('tar -xvf {} -C {}'.format(os.path.join(args.polyrnn2_dir, 'data', 'cityscapes.tar.gz'), os.path.join(args.polyrnn2_dir, 'data')))
    os.system('2to3 -w {}'.format(args.polyrnn2_dir))
    sys.path.append(os.path.join(args.polyrnn2_dir, 'code'))
    from DataProvider.cityscapes import DataProvider, collate_fn
    from Utils import utils
    
    main(args)