import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

import numpy as np; np.set_printoptions(precision=4)
import shutil, argparse, logging, sys, time, os
from tensorboardX import SummaryWriter

from loader import CityScapeLoader
from model import BatchedRasterLoss2D, PeriodicUpsample1D, PolygonNet

def initialize_logger(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    shutil.copy2(__file__, os.path.join(args.log_dir, "train.py"))
    shutil.copy2("model.py", os.path.join(args.log_dir, "model.py"))
    shutil.copy2("loader.py", os.path.join(args.log_dir, "loader.py"))
    shutil.copy2("run.sh", os.path.join(args.log_dir, "run.sh"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)
    logger.info("%s", repr(args))
    return logger

def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    if epoch > 0:
        os.remove(output_folder + filename + '_%03d' % (epoch-1) + '.pth.tar')
    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')
    if is_best:
        logger.info("Saving new best model")
        shutil.copyfile(output_folder + filename + '_%03d' % epoch + '.pth.tar',
                        output_folder + filename + '_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def masked_miou(output_raster, target_raster):
    """
    compute masked mean-IoU score by comparing output and target rasters
    :param output_raster: shape [N, res, res]
    :param target_raster: shape [N, res, res]
    """
    output_raster = torch.clamp(output_raster, 0, 1)
    target_raster = torch.clamp(target_raster, 0, 1)
    output_mask = (output_raster >= 0.5)
    target_mask = (target_raster >= 0.5)

    intersect = ((output_mask == 1) + (target_mask == 1)).eq(2).sum().item()
    union = ((output_mask == 1) + (target_mask == 1)).ge(1).sum().item()
    return intersect / union

def compose_masked_img(polygon_raster, input):
    """
    compose masked image for visualization
    :param polygon_raster: shape [N, res, res]
    :param input: shape [N, 3, res, res]
    """
    dtype = input.dtype
    device = input.device
    polyraster = torch.clamp(polygon_raster.float(), 0, 1)
    red_mask = (polyraster.unsqueeze(-1) * torch.tensor([1, 0, 0], dtype=dtype, device=device)).permute(0, 3, 1, 2)
    composed_img = torch.clamp(red_mask * 0.5 + input, 0, 1)
    return composed_img

def train(args, model, loader, criterion, optimizer, epoch, device, logger, writer):
    model.train()

    losses = AverageMeter()
    mious = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        n = input.size(0)
        input, target = input.to(device), target.to(device)
        output = model(input) # output shape [N, 128, 2]
        loss, output_raster = criterion(output, target)

        # measure miou and record loss
        miou = masked_miou(output_raster, target)
        losses.update(loss.item(), n)
        mious.update(miou, n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update statistics
        losses.update(loss, n)

        # update TensorboardX
        writer.add_scalar('training/loss', losses.avg, args.TRAIN_GLOB_STEP)
        writer.add_scalar('training/miou', mious.avg, args.TRAIN_GLOB_STEP)
        writer.add_scalar('training/batch_time', batch_time.avg, args.TRAIN_GLOB_STEP)
        writer.add_scalar('training/data_time', data_time.avg, args.TRAIN_GLOB_STEP)

        if batch_idx % args.log_interval == 0:
            log_text = ('Train Epoch: [{0}][{1}/{2}]\t'
                        'CompTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'mIoU {miou.val:.3f} ({miou.avg:.3f})\t').format(
                         epoch, batch_idx, len(loader), batch_time=batch_time,
                         data_time=data_time, loss=losses, miou=mious)
            logger.info(log_text)
            # update TensorboardX
            compimg = compose_masked_img(output_raster, input)
            imgrid = vutils.make_grid(compimg, normalize=False, scale_each=False)
            writer.add_image('training/predictions', imgrid, args.TRAIN_GLOB_STEP)
            writer.add_text('training/text_log', log_text, args.TRAIN_GLOB_STEP)

        args.TRAIN_GLOB_STEP += 1

def validate(args, model, loader, criterion, epoch, device, logger, writer):
    model.eval()

    losses = AverageMeter()
    mious = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            n = input.size(0)
            input, target = input.to(device), target.to(device)
            output = model(input) # output shape [N, 128, 2]
            loss, output_raster = criterion(output, target)

            # measure miou and record loss
            miou = masked_miou(output_raster, target)
            losses.update(loss.item(), n)
            mious.update(miou, n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update statistics
            losses.update(loss, n)

        log_text = ('Valid Epoch: [{0}]\t'
                    'CompTime {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                    'DataTime {data_time.sum:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.avg:.4f}\t'
                    'mIoU {miou.avg:.3f}\t').format(
                     epoch, batch_time=batch_time,
                     data_time=data_time, loss=losses, miou=mious)
        logger.info(log_text)

        # update TensorboardX
        compimg = compose_masked_img(output_raster, input)
        imgrid = vutils.make_grid(compimg, normalize=False, scale_each=False)
        writer.add_image('validation/predictions', imgrid, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/loss', losses.avg, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/miou', mious.avg, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/batch_time', batch_time.avg, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/data_time', data_time.avg, args.VAL_GLOB_STEP)
        writer.add_text('validation/text_log', log_text, args.VAL_GLOB_STEP)
        args.VAL_GLOB_STEP += 1

    return mious.avg

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=32, metavar='N',
                        help='input batch size for validation (default: 32)')
    parser.add_argument('--loss_type', type=str, choices=['l1', 'l2'], default='l1', help='type of loss for raster loss computation')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_folder', type=str, default="processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--resume', type=str, default=None, help="path to checkpoint if resume is needed")
    parser.add_argument('--timestamp', action='store_true', help="add timestamp to log_dir name")

    args = parser.parse_args()
    args.TRAIN_GLOB_STEP = 0
    args.VAL_GLOB_STEP = 0

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    # boiler-plate
    if args.timestamp:
        args.log_dir += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = initialize_logger(args)
    torch.manual_seed(args.seed)

    # TensorboardX writer
    args.tblogdir = os.path.join(args.log_dir, "tensorboard_log")
    if not os.path.exists(args.tblogdir):
        os.makedirs(args.tblogdir)
    writer = SummaryWriter(log_dir=args.tblogdir)

    # get training / valid sets
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # DO NOT include horizontal and vertical flips in the composed transforms!
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=.1, saturation=.1),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    trainset = CityScapeLoader(args.data_folder, "train", transforms=transform, RandomHorizontalFlip=0.5, RandomVerticalFlip=0.0)
    valset = CityScapeLoader(args.data_folder, "val", transforms=transform, RandomHorizontalFlip=0.0, RandomVerticalFlip=0.0)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)
    
    # initialize and parallelize model
    model = PolygonNet()
    model = nn.DataParallel(model)
    model.to(device)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.start_epoch = -1
    args.best_miou = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_miou = checkpoint['best_miou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    checkpoint_path = os.path.join(args.log_dir, 'checkpoint')
    criterion = BatchedRasterLoss2D(loss=args.loss_type, return_raster=True).to(device)

    # training loop
    for epoch in range(args.start_epoch + 1, args.epochs):
        if args.decay:
            scheduler.step(epoch)
        loss = train(args, model, train_loader, criterion, optimizer, epoch, device, logger, writer)
        miou = validate(args, model, val_loader, criterion, epoch, device, logger, writer)
        if miou > args.best_miou:
            args.best_miou = miou
            is_best = True
        else:
            is_best = False
        save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_miou': best_miou,
        'optimizer': optimizer.state_dict(),
        }, is_best, epoch, checkpoint_path, "_polygonnet", logger)

if __name__ == "__main__":
    main()