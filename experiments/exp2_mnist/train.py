import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import logging
import os
import argparse
import numpy as np
import datetime
import shutil
import multiprocessing

from loader import PMNISTDataSet
from model import LeNet5

MEAN = 0.14807655
STD = 0.36801067

def init_logger(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    shutil.copy2(__file__, os.path.join(args.logdir, "train.py"))
    shutil.copy2("model.py", os.path.join(args.logdir, "model.py"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.logdir, "log.txt"))
    logger.addHandler(fh)
    return logger

def train(args, model, optimizer, epoch, device, loader, logger):
    model.train()
    running_loss = 0
    c_train = 0
    count = 0
    for batch_idx, data in enumerate(loader):
        # get the inputs
        inputs, labels = data['input'], data['label']
        # wrap them in Variable
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        # compute accuracy
        _, predicted = torch.max(outputs.data, 1)

        # write statistics
        running_loss += loss.item()
        c_train += (predicted == labels.data).sum().item()
        count += inputs.shape[0]

        if batch_idx % args.log_interval == 0:
            logger.info('<{}> Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} ACCU: {:.6f}'.format(
                loader.dataset.partition,
                epoch, count, len(loader.dataset), 100. * batch_idx / len(loader), 
                loss.item(), c_train/count))

def test(args, model, epoch, device, loader, logger):
    model.eval()
    c_test = 0
    count = 0
    for batch_idx, data in enumerate(loader):
        # get the inputs
        inputs, labels = data['input'], data['label']
        # wrap them in Variable
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        # forward + backward + optimize
        outputs = model(inputs)
        # compute accuracy
        _, predicted = torch.max(outputs.data, 1)

        # write statistics
        c_test += (predicted == labels.data).sum().item()
        count += inputs.shape[0]

    logger.info('<{}> Epoch: {} ACCU: {:.6f}'.format(
        loader.dataset.partition, epoch, c_test/count))


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='MNIST experiment (training)')
    parser.add_argument('-s', '--imsize', type=int, default=28, help='input image size (resolution)')
    parser.add_argument('-l', '--logdir', type=str, help='log directory path')
    parser.add_argument('-d', '--datadir', type=str, default="./data/polyMNIST", help='data directory path')
    parser.add_argument('-n', '--epoch', type=int, default=20, help='number of epochs to train')
    parser.add_argument('-j', '--jobs', type=int, default=-1, help='number of threads to use')
    parser.add_argument('-b', '--batch', type=int, default=64, help='batch size to use')
    parser.add_argument('-c', '--cache_root', type=str, default='./cache', help='path to directory to cache')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--no_cuda', action='store_true', default=0, help='Not use GPU.')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'],metavar='OPT', help='optimizer')
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval')
    parser.add_argument('--no_decay', action='store_true', help='decay learning rate')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.logdir is None:
        dt=datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        args.logdir = "logs/log-"+dt
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.jobs < 0:
        args.jobs = multiprocessing.cpu_count()

    # logger and snapshot current code
    logger = init_logger(args)
    logger.info("%s", repr(args))


    # dataloader
    trainset = PMNISTDataSet(args.datadir, 'train', imsize=args.imsize, cache_root=args.cache_root)
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=args.jobs)
    testset = PMNISTDataSet(args.datadir, 'test', imsize=args.imsize, cache_root=args.cache_root)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=True, num_workers=args.jobs)

    model = LeNet5((args.imsize, args.imsize), MEAN, STD)
    model = model.double()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # LR decay scheduler
    if not args.no_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if use_cuda:
        model = nn.DataParallel(model)

    model.to(device)

    for epoch in range(args.epoch):  # loop over the dataset
        train(args, model, optimizer, epoch, device, trainloader, logger)
        test(args, model, epoch, device, testloader, logger)

    # save checkpoint
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, os.path.join(args.logdir, 'checkpoint_final.pth.tar'))

if __name__ == '__main__':
    main()
