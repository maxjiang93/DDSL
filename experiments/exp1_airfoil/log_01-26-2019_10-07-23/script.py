from AFNet import *
from airfoil_dataset import *

from torch.autograd import Variable
import torch.optim as optim

from tensorboardX import SummaryWriter

import shutil
from shutil import rmtree

import argparse
import logging

import datetime

from sklearn.metrics import r2_score

# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)

# Define mean squared error function
def MSE(predicted, labels):
    return torch.mean((predicted-labels)**2, dim=0)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train AFNet')
    parser.add_argument('--bottleneck', type=int, default=32,
                        help='number of channels out of ResNet-50 block in AFNet (default:10)')
    parser.add_argument('--out_ch', type=int, default=1,
                        help='number of channels out of AFNet, i.e. number of training objectives (default:2)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--shape-dir', type=str, default='processed_data',
                        help='path to shapes folder (default: processed_data)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-file', type=str, default="processed_data/airfoil_data_normalized.csv",
                        help='data file containing preprocessed airfoil data (Re, Cl, Cd, names)')
    parser.add_argument('--log-dir', type=str, default="", 
                        help='log directory for run (default: log_<time>)')
    parser.add_argument('--tb-log-dir', type=str, default="",
                        help='log directory for Tensorboard, defaulted to same directory as logger (default: log_<date>)')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--dropout', action="store_true")
    parser.add_argument('--checkpoint', default="",
                        help='uses given checkpoint file')

    args = parser.parse_args()
    
    # Clear cuda gpu cache
    torch.cuda.empty_cache()

    # Make log directory and files
    if args.log_dir=="":
        dt=datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        log_dir='log_'+dt
    else:
        log_dir=args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    shutil.copy2(__file__, os.path.join(log_dir, "script.py"))
    shutil.copy2("AFNet.py", os.path.join(log_dir, "AFNet.py"))
    shutil.copy2("run.sh", os.path.join(log_dir, "run.sh"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(args))

    # Check if use cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Set device to use (cuda vs cpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print('\nUsing Cuda.\n')

    # Set random seed
    torch.manual_seed(args.seed)

    # Create datasets and dataloaders
    print('\nLoading data...')
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    trainset = AirfoilDataset(csv_file=args.data_file, shape_dir=args.shape_dir, set_type='train')
    validset = AirfoilDataset(csv_file=args.data_file, shape_dir=args.shape_dir, set_type='valid')
#     testset = AirfoilDataset(csv_file=args.data_file, shape_dir=args.shape_dir, set_type='test')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,**kwargs)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True, **kwargs)
#     testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, set_type='test', **kwargs)
    print('\nTraining on '+str(len(trainset))+' data points.')
    print('Validating on '+str(len(validset))+' data points.\n')
    
    # Get model
    net = AFNet(bottleneck=args.bottleneck, out_ch=args.out_ch)
    net = net.double()
    if(torch.cuda.device_count()>1):
        print('\nUsing', torch.cuda.device_count(), 'GPUs.')
        net=nn.DataParallel(net)
    net.to(device)

    # Log number of parameters
    logger.info("{} parameters in total".format(sum(x.numel() for x in net.parameters())))

    # Set optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    # Get checkpoint
    if args.checkpoint!="":
        checkpoint=torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        print('Using checkpoint '+args.checkpoint)
        
    # Mean squared error loss
    criterion=nn.MSELoss()
        
    # Learning rate decay
    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Start Tensorboard
    if args.tb_log_dir=="":
        tb_log_dir=log_dir
    else:
        tb_log_dir=args.tb_log_dir
    if not os.path.exists(tb_log_dir):
        rmtree(tb_log_dir)
    writer = SummaryWriter(tb_log_dir)
    print('\nTensorboard initiated.')
    print('\nTraining neural network...')
    
    # Train AFNet
    for epoch in range(args.epochs):
        if args.decay:
            scheduler.step()
            
        running_loss=0.0
        valid_running_loss=0.0
        acc_valid=0.0
        acc_train=0.0
        for i, data in enumerate(validloader, 0):
            # Get inputs
            shapes=data['shape']
            shapes=shapes.to(device)
            cfd_data=torch.cat((data['Re'].view(-1, 1), data['aoa'].view(-1, 1)), 1)
            cfd_data=cfd_data.to(device)
            
            # Get labels
            labels=data['Cl/Cd'].view(-1, 1).to(device)
            labels=labels.to(device)

            # Forward
            outputs=net(shapes, cfd_data)
            loss=criterion(outputs, labels)

            # Running loss
            valid_running_loss+=loss.item()

            # Accuracy
            r2=r2_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            acc_valid+=r2

            # Write to log
            if i%args.log_interval==0:
                logger.info('Validation set [{}/{} ({:.0f}%)]: Loss: {:.4f}, Cl/Cd R2 score: {:.4f}\r'.format(i*len(shapes), len(validloader.dataset), 100.*i*len(shapes)/len(validloader.dataset), loss.item(), r2))
                
        for i, data in enumerate(trainloader, 0):
            # Get inputs
            shapes=data['shape']
            shapes=shapes.to(device)
            cfd_data=torch.cat((data['Re'].view(-1, 1), data['aoa'].view(-1, 1)), 1)
            cfd_data=cfd_data.to(device)

            # Get labels
            labels=data['Cl/Cd'].view(-1, 1).to(device)
            labels=labels.to(device)
        
            # Zero gradients
            optimizer.zero_grad()

            # Forward
            outputs=net(shapes, cfd_data)
            loss=criterion(outputs, labels)

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # Accuracy
            r2=r2_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            acc_train+=r2

            # Running loss
            running_loss+=loss.item()
            
            # Write to log
            if i%args.log_interval==0:
                logger.info('Train set [{}/{} ({:.0f}%)]: Loss: {:.4f}, Cl/Cd R2 score: {:.4f}\r'.format(i*len(shapes), len(trainloader.dataset), 100.*i*len(shapes)/len(trainloader.dataset), loss.item(), r2))
            
        # Write statistics to Tensorboard
        avg_acc_train=acc_train/len(trainloader)
        avg_acc_valid=acc_valid/len(validloader)
        train_loss=running_loss/len(trainloader)
        valid_loss=valid_running_loss/len(validloader)
        writer.add_scalars('data/accuracy', {'valid_acc': avg_acc_valid,\
                                            'train_acc': avg_acc_train}, epoch)
        writer.add_scalars('data/loss', {'train_loss':train_loss,\
                                         'valid_loss':valid_loss}, epoch)
        
        # Write to log
        logger.info('[Epoch {}] Train set: Average loss: {:.4f}, Avg Cl/Cd R2 score: {:.4f}\r'.format(epoch, train_loss, avg_acc_train))
        logger.info('[Epoch {}] Validation set: Average loss: {:.4f}, Avg Cl/Cd R2 score: {:.4f}\r'.format(epoch, valid_loss, avg_acc_valid))

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model': net.state_dict(),
            'optim':optimizer.state_dict()
        }, os.path.join(log_dir,'AFNet_model.checkpoint'))

        # Zero statistics
        acc_valid=0.0
        acc_train=0.0
        running_loss = 0.0
        valid_running_loss = 0.0

    print('\nFinished training!')

    # Save model
    torch.save({
        'epoch': epoch,
        'model': net.state_dict(),
        'optim':optimizer.state_dict()
    }, os.path.join(log_dir,'AFNet_model'))

if __name__ == "__main__":
    main()
