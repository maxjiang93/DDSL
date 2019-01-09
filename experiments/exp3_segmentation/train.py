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
from model import BatchedRasterLoss2D, PeriodicUpsample1D, PolygonNet, PolygonNet2

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
        self.n = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.n = n
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def valcavg(self):
        return self.val.sum().item() / (self.n != 0).sum().item()

    @property
    def avgcavg(self):
        return self.avg.sum().item() / (self.count != 0).sum().item()
    

def masked_miou(output_raster, target_raster, labels, nclass):
    """
    compute masked mean-IoU score by comparing output and target rasters
    :param output_raster: shape [N, res, res]
    :param target_raster: shape [N, res, res]
    :param labels: shape [N, res, res]
    :param nclass: int, number of classes
    :return [nclass] float for miou corresponding to each class
            [nclass] int for number of counts for per class
    """
    device = output_raster.device

    output_raster = torch.clamp(output_raster, 0, 1)
    target_raster = torch.clamp(target_raster, 0, 1)
    output_mask = (output_raster >= 0.5)
    target_mask = (target_raster >= 0.5)

    intersect = ((output_mask == 1) + (target_mask == 1)).eq(2).sum((1,2))
    union = ((output_mask == 1) + (target_mask == 1)).ge(1).sum((1,2))
    iou_all = (intersect.float() / union.float()).unsqueeze(-1) # [N]
    # get per class miou score
    labels_one_hot = torch.arange(nclass, device=device).expand(labels.shape[0], nclass) == labels.unsqueeze(-1)
    labels_one_hot = labels_one_hot.float()
    count_per_cls = labels_one_hot.sum(0)

    miou_per_cls = torch.sum(labels_one_hot * iou_all, dim=0) / count_per_cls
    miou_per_cls[torch.isnan(miou_per_cls)] = 0

    return miou_per_cls, count_per_cls

def compose_masked_img(polygon_raster, input, mean, std):
    """
    compose masked image for visualization
    :param polygon_raster: shape [N, res, res]
    :param input: shape [N, 3, res, res]
    :param mean: mean for denormalizing image
    :param std: std for denormalizing image
    """
    dtype = input.dtype
    device = input.device
    img_mean = torch.tensor(mean, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
    img_std  = torch.tensor(std , dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
    input = (input * img_std ) + img_mean
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
    for batch_idx, (input, target, label) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        n = input.size(0)
        input, target, label = input.to(device), target.to(device), label.to(device)
        output = model(input) # output shape [N, 128, 2]
        loss_vec, output_raster = criterion(output, target)
        loss = (loss_vec * args.weights[label]).sum()

        # measure miou and record loss
        miou, miou_count = masked_miou(output_raster, target, label, args.nclass)
        losses.update(loss.item(), n)
        mious.update(miou, miou_count)

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
        writer.add_scalar('training/loss', losses.val, args.TRAIN_GLOB_STEP)
        writer.add_scalar('training/batch_time', batch_time.val, args.TRAIN_GLOB_STEP)
        writer.add_scalar('training/data_time', data_time.val, args.TRAIN_GLOB_STEP)
        writer.add_scalar('training/miou_mean', mious.valcavg, args.TRAIN_GLOB_STEP)

        if args.TRAIN_GLOB_STEP % args.log_interval == 0:
            log_text = ('Train Epoch: [{0}][{1}/{2}]\t'
                        'CompTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'mIoU {miou:.3f} ({miou:.3f})\t').format(
                         epoch, batch_idx, len(loader), batch_time=batch_time,
                         data_time=data_time, loss=losses, miou=mious.val.mean().item())
            logger.info(log_text)
            # update TensorboardX
            compimg = compose_masked_img(output_raster, input, args.img_mean, args.img_std)
            imgrid = vutils.make_grid(compimg, normalize=True, scale_each=True)
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
        for batch_idx, (input, target, label) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            n = input.size(0)
            input, target, label = input.to(device), target.to(device), label.to(device)
            output = model(input) # output shape [N, 128, 2]
            loss_vec, output_raster = criterion(output, target)
            loss = (loss_vec * args.weights[label]).sum()

            # measure miou and record loss
            miou, miou_count = masked_miou(output_raster, target, label, args.nclass)
            losses.update(loss.item(), n)
            mious.update(miou, miou_count)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update statistics
            losses.update(loss, n)

        log_text = ('Valid Epoch: [{0}]\t'
                    'CompTime {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                    'DataTime {data_time.sum:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.avg:.4f}\t'
                    'mIoU {miou:.3f}\t').format(
                     epoch, batch_time=batch_time,
                     data_time=data_time, loss=losses, miou=mious.avgcavg)
        logger.info(log_text)

        # update TensorboardX
        compimg = compose_masked_img(output_raster, input, args.img_mean, args.img_std)
        imgrid = vutils.make_grid(compimg, normalize=False, scale_each=False)
        writer.add_image('validation/predictions', imgrid, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/loss', losses.avg, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/miou', mious.avgcavg, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/batch_time', batch_time.avg, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/data_time', data_time.avg, args.VAL_GLOB_STEP)
        writer.add_scalar('validation/miou_mean', mious.avg.mean().item(), args.VAL_GLOB_STEP)
        # update per-class miou
        for i, name  in enumerate(args.label_names):
            writer.add_scalar('validation/miou_' + name, mious.avg[i].item(), args.VAL_GLOB_STEP)
        writer.add_text('validation/text_log', log_text, args.VAL_GLOB_STEP)
        args.VAL_GLOB_STEP += 1

    return mious.avgcavg

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
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args.TRAIN_GLOB_STEP = 0
    args.VAL_GLOB_STEP = 0
    args.label_names = ["car", "truck", "train", "bus", "motorcycle", "bicycle", "rider", "person"]
    args.instances = torch.tensor([30246, 516, 76, 325, 658, 3307, 1872, 19563]).double().to(device)
    args.nclass = len(args.label_names)
    args.weights = 1 / torch.log(args.instances / args.instances.sum() + 1.02)
    args.weights /= args.weights.sum()

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    # boiler-plate
    if args.timestamp:
        args.log_dir += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = initialize_logger(args)
    torch.manual_seed(args.seed)

    # TensorboardX writer
    args.tblogdir = os.path.join(args.log_dir, "tensorboard_log")
    if not os.path.exists(args.tblogdir):
        os.makedirs(args.tblogdir)
    writer = SummaryWriter(log_dir=args.tblogdir)

    # get training / valid sets
    args.img_mean = [0.485, 0.456, 0.406]
    args.img_std = [0.229, 0.224, 0.225]
    normalize = torchvision.transforms.Normalize(mean=args.img_mean,
                                                  std=args.img_std)
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
    # model = PolygonNet()
    model = PolygonNet2()
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
    criterion = BatchedRasterLoss2D(npoints=model.module.npoints, loss=args.loss_type, return_raster=True).to(device)

    # training loop
    for epoch in range(args.start_epoch + 1, args.epochs):
        if args.decay:
            scheduler.step(epoch)
        train(args, model, train_loader, criterion, optimizer, epoch, device, logger, writer)
        miou = validate(args, model, val_loader, criterion, epoch, device, logger, writer)
        if miou > args.best_miou:
            args.best_miou = miou
            is_best = True
        else:
            is_best = False
        save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_miou': args.best_miou,
        'optimizer': optimizer.state_dict(),
        }, is_best, epoch, checkpoint_path, "_polygonnet", logger)

if __name__ == "__main__":
    main()