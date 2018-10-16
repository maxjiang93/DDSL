from AFNet import *
from AFNet_dataset import *

from torch.autograd import Variable
import torch.optim as optim

from tensorboardX import SummaryWriter

import shutil
from shutil import rmtree

import argparse
import logging

import datetime

# Define mean squared error function
def MSE(predicted, labels):
    return torch.mean((predicted-labels)**2, dim=0)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train AFNet')
    parser.add_argument('--bottleneck', type=int, default=10,
                        help='number of channels out of ResNet-50 block in AFNet (default:10)')
    parser.add_argument('--out_ch', type=int, default=2,
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
    parser.add_argument('--shape_dir', type=str, default='processed-data',
                        help='path to shapes folder (default: processed-data)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data_file', type=str, default="processed-data/airfoil_data.csv",
                        help='data file containing preprocessed airfoil data (Re, Cl, Cd, names)')
    parser.add_argument('--log_dir', type=str, default="", 
                        help='log directory for run (default: log_<time>)')
    parser.add_argument('--tb_log_dir', type=str, default="",
                        help='log directory for Tensorboard, defaulted to same directory as logger (default: log_<date>)')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--dropout', action="store_true")
    parser.add_argument('--checkpoint', default="",
                        help='uses given checkpoint file')

    args = parser.parse_args()

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
#     shutil.copy2("run.sh", os.path.join(log_ssh davdir, "run.sh"))

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

    # Set random seed
    torch.manual_seed(args.seed)

    # Create datasets and dataloaders
    print('\nLoading data...')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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
    if(torch.cuda.device_count()>1):
        print('\nUsing', torch.cuda.device_count(), 'GPUs.')
        net=nn.DataParallel(net)
    net.to(device)
    
    # Get checkpoint
    if args.checkpoint!="":
        model=torch.load(args.checkpoint)
        net.load_state_dict(model['model'])
        print('Using checkpoint '+args.checkpoint)

    # Log number of parameters
    logger.info("{} parameters in total".format(sum(x.numel() for x in net.parameters())))

    # Set optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        
    # Mean squared error loss
    criterion=nn.MSELoss()
        
    # Learning rate decay
    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
        running_loss=0
        valid_running_loss=0
        acc_valid=torch.zeros(2).float().to(device)
        acc_train=torch.zeros(2).float().to(device)
        for i, data in enumerate(validloader, 0):
            # Get inputs
            shapes=data['shape']
            shapes=torch.Tensor(shapes)
            shapes=shapes.float().to(device)
            cfd_data=data['Re'].view(-1, 1).float().to(device)

            # Get labels
            labels=torch.cat((data['Cl'].view(-1, 1), data['Cd'].view(-1, 1)), 1).float()
            labels=labels.to(device)

            # Forward
            outputs=net(shapes, cfd_data)
            loss=criterion(outputs, labels)

            # Running loss
            valid_running_loss+=loss.data.item()

            # Accuracy
            acc_valid+=MSE(outputs.data, labels.data)

        for i, data in enumerate(trainloader, 0):
            # Get inputs
            shapes=data['shape']
            shapes=torch.Tensor(shapes)
            shapes=shapes.float().to(device)
            cfd_data=data['Re'].view(-1, 1).float().to(device)

            # Get labels
            labels=torch.cat((data['Cl'].view(-1, 1), data['Cd'].view(-1, 1)), 1).float()
            labels=labels.to(device)

            # Forward
            outputs=net(shapes, cfd_data)
            loss=criterion(outputs, labels)

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # Accuracy
            acc_train+=MSE(outputs.data, labels.data)

            # Running loss
            running_loss+=loss.data.item()

        # Write statistics to Tensorboard
        avg_acc_valid=acc_train/len(validloader)
        avg_acc_train=acc_train/len(trainloader)
        train_loss=running_loss/len(trainloader.dataset)
        valid_loss=valid_running_loss/len(validloader.dataset)
        writer.add_scalars('data/accuracy', {'valid_cl_acc': avg_acc_valid[0],\
                                            'valid_cd_acc': avg_acc_valid[1],\
                                            'train_cl_acc': avg_acc_train[0],\
                                            'train_cd_acc': avg_acc_train[1]}, epoch)
        writer.add_scalars('data/loss', {'train_loss':train_loss,\
                                         'valid_loss':valid_loss}, epoch)
        
        # Write to log
        logger.info('Train set: Average loss: {:.4f}\r'.format(
            train_loss))
        logger.info('Validation set: Average loss: {:.4f}\r'.format(
            valid_loss))

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model': net.state_dict(),
            'optim':optimizer.state_dict()
        }, os.path.join(log_dir,'AFNet_model.checkpoint'))

        # Zero statistics
        acc_valid=torch.zeros(2).float().to(device)
        acc_train=torch.zeros(2).float().to(device)
        running_loss = 0.0
        valid_running_loss = 0.0

        if epoch%5==0:
            print('Epoch '+str(epoch)+' complete!')

    print('\nFinished training!')

    # Save model
    torch.save({
        'epoch': epoch,
        'model': net.state_dict(),
        'optim':optimizer.state_dict()
    }, os.path.join(log_dir,'AFNet_model'))

if __name__ == "__main__":
    main()