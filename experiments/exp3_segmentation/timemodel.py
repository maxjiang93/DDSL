import torch
import torch.nn as nn
import torchvision
from loader import CityScapeLoader
from torch.utils.data import Dataset, DataLoader
from model import PolygonNet
import numpy as np

import argparse, time, os
from tqdm import tqdm

import sys; sys.path.append("baseline")


def evaluate(args, model, loader, device):
    model.eval()
    time_ = []
    count = 0

    pbar = tqdm(total=20)

    with torch.no_grad():
        for batch_idx, (input, _, _) in enumerate(loader):
            # compute output
            num = input.size(0)
            input = input.to(device)
            t0 = time.time()
            output = model(input)
            t1 = time.time()
            if count > 0: # discard first batch time
                time_.append(t1 - t0)
                pbar.update(1)
            if count >= 20: # take the average of first 20 batch
                break
            count += 1
    pbar.close()
    times = np.array(time_)
    avg = np.mean(times)
    std = np.std(times)
    print("Per Batch Time AVG: {}, STDEV: {}".format(avg, std))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                        help='input batch size for test (default: 64)')
    parser.add_argument('--nlevels', type=int, default=4, help="number of polygon levels, higher->finer")
    parser.add_argument('--feat', type=int, default=256, help="number of base feature layers")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_folder', type=str, default="mres_processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/checkpoint_polygonnet_best.pth.tar', help="path to checkpoint to load")
    parser.add_argument('--transpose', action='store_true', help="transpose target")
    parser.add_argument('--workers', default=12, type=int, help="number of data loading workers")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    torch.manual_seed(args.seed)

    # get training / valid sets
    args.img_mean = [0.485, 0.456, 0.406]
    args.img_std = [0.229, 0.224, 0.225]
    normalize = torchvision.transforms.Normalize(mean=args.img_mean,
                                                  std=args.img_std)
    # DO NOT include horizontal and vertical flips in the composed transforms!
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    testset = CityScapeLoader(args.data_folder, "test", transforms=transform, RandomHorizontalFlip=0.0, RandomVerticalFlip=0.0, mres=False, transpose=args.transpose)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, drop_last=False, num_workers=args.workers, pin_memory=True)

    # initialize model
    model = PolygonNet(nlevels=args.nlevels, dropout=False, feat=args.feat)
    model = nn.DataParallel(model)
    if os.path.isfile(args.ckpt):
        print("=> loading checkpoint '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        args.best_miou = checkpoint['best_miou']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.ckpt))
    model.to(device)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    evaluate(args, model, test_loader, device)

if __name__ == '__main__':
    main()
