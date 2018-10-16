import resnet
from resnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# for debugging
class MyConv(nn.Module):
    def __init__(self):
        super(MyConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)

    def forward(self, x, y):
        # do something
        # Pad shape
        x=torch.cat((x,torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3]).cuda()), dim=2)
        
        # Take irfft of input shape
        x=torch.irfft(x, 2, signal_sizes=[224,224])
        print(x.size())
        
        # Resize shape into 4D tensor for input
        x=x.view(-1, 1, 224, 224)
        
        x=self.conv1(x)
        return x
        

# Define convolutional neural network
class AFNet(nn.Module):
    def __init__(self, bottleneck=10, out_ch=2):
        super(AFNet, self).__init__()
        self.bottleneck=bottleneck
        self.out_ch=out_ch
        self.resnet=resnet50(num_classes=bottleneck)
        self.fc=nn.Linear(bottleneck+1,out_ch)
    
    def forward(self, shapes, cfd_data): 
        # Pad shape
        x=torch.cat((shapes,torch.zeros(shapes.shape[0], shapes.shape[1], 1, shapes.shape[3]).cuda()), dim=2)
        
        # Take irfft of input shape
        x=torch.irfft(x, 2, signal_sizes=[224,224])
        
        # Resize shape into 4D tensor for input
        x=x.view(-1, 1, 224, 224)
        
        # Go through Resnet
        x=F.relu(self.resnet(x))
#         x=self.resnet(x)
        
        # Append CFD data
        x=self.append_cfd_data(x, cfd_data)
        
        # Fully connected layer
        x=self.fc(x)

        return x

    def append_cfd_data(self, x, cfd_data):
        # Append data
        x=torch.cat((x.data, cfd_data.data), 1)
        # x=x.cuda()
        return x
    