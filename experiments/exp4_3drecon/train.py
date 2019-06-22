import torch
import torchvision
import torch.optim as optim 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

import numpy as np; np.set_printoptions(precision=4)
import shutil, argparse, logging, sys, time, os, pickle, trimesh
from tensorboardX import SummaryWriter
from collections import OrderedDict

from loader import ShapeNetLoader
from model import SphereNet, BatchedRasterLoss3D, ChamferLoss, MeshSampler

torch.backends.cudnn.benchmark=True


def load_my_state_dict(self, state_dict, exclude='none'):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if exclude in name:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def save_meshsamp(args, v, epoch):
    """
    Args:
      v: [N, 642, 3]
    """
    b = v.shape[0]
    f = args.faces.detach().cpu().numpy()
    outdir = os.path.join(args.meshsamp_dir, "ep_{:04d}".format(epoch))
    if not os.path.exists(outdir): os.makedirs(outdir)
    for i in range(b):
        v_ = v[i]
        mesh = trimesh.Trimesh(v_, f)
        fname = os.path.join(outdir, "sample_{:02d}.obj".format(i))
        mesh.export(fname)


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
        prev_file = output_folder + filename + '_%03d' % (epoch-1) + '.pth.tar'
        if os.path.exists(prev_file):
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


def train(args, model, loader, criterion_r, criterion_c, mesh_sampler, optimizer, epoch, device, logger, writer):
    model.train()
    
    losses_sum = AverageMeter()
    losses_cham = AverageMeter()
    losses_accu = AverageMeter()
    losses_comp = AverageMeter()
    losses_rast = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    time0 = time.time()

    end = time.time()
    for batch_idx, (img, raster_gt, pts_gt) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        num = img.size(0)
        img, raster_gt, pts_gt = (img.to(device), 
                                  raster_gt.to(device), 
                                  pts_gt.to(device))

        output = model(img) # output shape [N, 642, 3]
        pts_gen = mesh_sampler(output)  # [N, #s, 3]

        loss_raster, _ = criterion_r(output, raster_gt)
        loss_accu, loss_comp, loss_chamfer = criterion_c(pts_gen, pts_gt)

        loss_raster = torch.mean(loss_raster) # take mean of loss on multi gpus
        loss_accu = torch.mean(loss_accu)
        loss_comp = torch.mean(loss_comp)
        loss_chamfer = torch.mean(loss_chamfer)

        loss = args.alpha_chamfer * loss_chamfer + args.alpha_raster * loss_raster

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update statistics
        losses_accu.update(loss_accu.item(), num)
        losses_comp.update(loss_comp.item(), num)
        losses_cham.update(loss_chamfer.item(), num)
        losses_rast.update(loss_raster.item(), num)
        losses_sum.update(loss.item(), num)

        if args.TRAIN_GLOB_STEP % args.log_interval == 0:
            log_text = ('Train Epoch: [{0}][{1}/{2}]\t'
                        'CompTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'AccuLoss {ac.val:.4f} ({ac.avg:.4f})\t'
                        'CompLoss {cp.val:.4f} ({cp.avg:.4f})\t'
                        'ChamferLoss {cf.val:.4f} ({cf.avg:.4f})\t'
                        'RasterLoss {rt.val:.4f} ({rt.avg:.4f})\t'
                        'TotLoss {tl.val:.4f} ({tl.avg:.4f})\t').format(
                         epoch, batch_idx, len(loader), batch_time=batch_time,
                         data_time=data_time, ac=losses_accu, cp=losses_comp, 
                         cf=losses_cham, rt=losses_rast, tl=losses_sum)
            logger.info(log_text)
            # update TensorboardX
            writer.add_text('training/text_log', log_text, args.TRAIN_GLOB_STEP)
            writer.add_scalar('training/batch_time', batch_time.val, args.TRAIN_GLOB_STEP)
            writer.add_scalar('training/data_time', data_time.val, args.TRAIN_GLOB_STEP)
            writer.add_scalar('training/rasterloss', losses_rast.val, args.TRAIN_GLOB_STEP)
            writer.add_scalar('training/accuracy', losses_accu.val, args.TRAIN_GLOB_STEP)
            writer.add_scalar('training/complete', losses_comp.val, args.TRAIN_GLOB_STEP)
            writer.add_scalar('training/chamfer', losses_cham.val, args.TRAIN_GLOB_STEP)
            writer.add_scalar('training/totalloss', losses_sum.val, args.TRAIN_GLOB_STEP)

        args.TRAIN_GLOB_STEP += 1

    logger.info("Train Time per Epoch: {} min".format((time.time() - time0)/60))

def eval(args, model, loader, criterion_r, criterion_c, mesh_sampler, optimizer, epoch, device, logger, writer):
    model.eval()
    
    losses_sum = AverageMeter()
    losses_cham = AverageMeter()
    losses_accu = AverageMeter()
    losses_comp = AverageMeter()
    losses_rast = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    time0 = time.time()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (img, raster_gt, pts_gt) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            num = img.size(0)
            img, raster_gt, pts_gt = (img.to(device), 
                                      raster_gt.to(device), 
                                      pts_gt.to(device))

            output = model(img) # output shape [N, 642, 3]
            pts_gen = mesh_sampler(output)  # [N, #s, 3]

            loss_raster, _ = criterion_r(output, raster_gt)
            loss_accu, loss_comp, loss_chamfer = criterion_c(pts_gen, pts_gt)

            loss_raster = torch.mean(loss_raster) # take mean of loss on multi gpus
            loss_accu = torch.mean(loss_accu)
            loss_comp = torch.mean(loss_comp)
            loss_chamfer = torch.mean(loss_chamfer)

            loss = args.alpha_chamfer * loss_chamfer + args.alpha_raster * loss_raster

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # save mesh
            if batch_idx == 0:
                save_meshsamp(args, output.detach().cpu().numpy(), epoch)

            # update statistics
            losses_accu.update(loss_accu.item(), num)
            losses_comp.update(loss_comp.item(), num)
            losses_cham.update(loss_chamfer.item(), num)
            losses_rast.update(loss_raster.item(), num)
            losses_sum.update(loss.item(), num)

    log_text = ('Train Epoch: [{0}][{1}/{2}]\t'
                'CompTime {batch_time.avg:.3f}\t'
                'DataTime {data_time.avg:.3f}\t'
                'AccuLoss {ac.avg:.4f}\t'
                'CompLoss {cp.avg:.4f}\t'
                'ChamferLoss {cf.avg:.4f}\t'
                'RasterLoss {rt.avg:.4f}\t'
                'TotLoss {tl.avg:.4f}\t').format(
                 epoch, batch_idx, len(loader), batch_time=batch_time,
                 data_time=data_time, ac=losses_accu, cp=losses_comp, 
                 cf=losses_cham, rt=losses_rast, tl=losses_sum)
    logger.info(log_text)
    # update TensorboardX
    writer.add_text('valid/text_log', log_text, args.TRAIN_GLOB_STEP)
    writer.add_scalar('valid/batch_time', batch_time.val, args.TRAIN_GLOB_STEP)
    writer.add_scalar('valid/data_time', data_time.val, args.TRAIN_GLOB_STEP)
    writer.add_scalar('valid/rasterloss', losses_rast.val, args.TRAIN_GLOB_STEP)
    writer.add_scalar('valid/accuracy', losses_accu.val, args.TRAIN_GLOB_STEP)
    writer.add_scalar('valid/complete', losses_comp.val, args.TRAIN_GLOB_STEP)
    writer.add_scalar('valid/chamfer', losses_cham.val, args.TRAIN_GLOB_STEP)
    writer.add_scalar('valid/totalloss', losses_sum.val, args.TRAIN_GLOB_STEP)

    writer.add_scalar('epoch', epoch, args.TRAIN_GLOB_STEP)
    logger.info("Eval Time per Epoch: {} min".format((time.time() - time0)/60))

    return losses_cham.avg

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Geometry Reconstruction')
    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--loss_type', type=str, choices=['l1', 'l2'], default='l1', help='type of loss for raster loss computation')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--nlevels', type=int, default=3, help="number of polygon levels, higher->finer")
    parser.add_argument('--feat', type=int, default=256, help="number of base feature layers")
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--alpha_raster', type=float, default=1.0, help="scaling factor for raster loss")
    parser.add_argument('--alpha_chamfer', type=float, default=1.0, help="scaling factor for chamfer loss")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_folder', type=str, default="data",
                        help='path to data folder (default: data)')
    parser.add_argument('--mesh_folder', type=str, default="mesh_files",
                        help='path to mesh folder (default: mesh_files)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--resume', type=str, default=None, help="path to checkpoint if resume is needed")
    parser.add_argument('--timestamp', action='store_true', help="add timestamp to log_dir name")
    parser.add_argument('--workers', default=12, type=int, help="number of data loading workers")
    parser.add_argument('--n_tgt_pts', default=2048, type=int, help="number of target points per shape")
    parser.add_argument('--n_gen_pts', default=2048, type=int, help="number of points to sample per generated shape")
    parser.add_argument('--no_deform', action='store_true', default=False, help="do not predict mesh deformation")
    parser.add_argument('--model2', action='store_true', default=False, help="use alternative model")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.deform = not args.no_deform

    args.TRAIN_GLOB_STEP = 0

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

    trainset = ShapeNetLoader(args.data_folder, "train", npts=args.n_tgt_pts)
    valset = ShapeNetLoader(args.data_folder, "val", npts=args.n_tgt_pts)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.workers, pin_memory=True)

    # create log dir to save mesh
    args.meshsamp_dir = os.path.join(args.log_dir, "mesh_samples")
    if not os.path.exists(args.meshsamp_dir): os.makedirs(args.meshsamp_dir)
    
    # initialize and parallelize model
    if not args.model2:
        model = SphereNet(mesh_folder=args.mesh_folder, nlevels=args.nlevels, feat=args.feat, deform=args.deform)
    else:
        model = SphereNet(mesh_folder=args.mesh_folder, nlevels=args.nlevels, feat=args.feat)

    model = nn.DataParallel(model)
    model.to(device)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.start_epoch = -1
    args.best_chamfer = np.inf
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_chamfer = checkpoint['best_chamfer']
            load_my_state_dict(model, checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.90)

    checkpoint_path = os.path.join(args.log_dir, 'checkpoint')

    # define training criteria
    meshfile = 'icosphere_{}.pkl'.format(args.nlevels)
    with open(os.path.join(args.mesh_folder, meshfile), "rb") as fp: 
        pkl = pickle.load(fp)
    F = torch.tensor(pkl['F']).to(device)
    args.faces = F
    criterion_c = nn.DataParallel(ChamferLoss(reduction='mean')).to(device)
    criterion_r = nn.DataParallel(BatchedRasterLoss3D(F, loss=args.loss_type)).to(device)
    mesh_sampler = nn.DataParallel(MeshSampler(F, nsamp=args.n_gen_pts)).to(device)

    # training loop
    for epoch in range(args.start_epoch + 1, args.epochs):
        if args.decay:
            scheduler.step(epoch)
        train(args, model, train_loader, criterion_r, criterion_c, mesh_sampler, optimizer, epoch, device, logger, writer)
        eval_cham = eval(args, model, val_loader, criterion_r, criterion_c, mesh_sampler, optimizer, epoch, device, logger, writer)
        if eval_cham > args.best_chamfer:
            args.best_chamfer = eval_cham
            is_best = True
        else:
            is_best = False

        # remove sparse matrices since they cannot be stored
        state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() != "torch.cuda.sparse.FloatTensor"]
        state_dict_no_sparse = OrderedDict(state_dict_no_sparse)

        save_checkpoint({
        'epoch': epoch,
        'state_dict': state_dict_no_sparse,
        'best_chamfer': args.best_chamfer,
        'optimizer': optimizer.state_dict(),
        }, is_best, epoch, checkpoint_path, "_spherenet", logger)

if __name__ == "__main__":
    main()
