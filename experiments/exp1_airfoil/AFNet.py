import resnet
from resnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable        

# Define convolutional neural network
class AFNet(nn.Module):
    def __init__(self, bottleneck=10, out_ch=1):
        super(AFNet, self).__init__()
        self.bottleneck=bottleneck
        self.out_ch=out_ch
        self.resnet=resnet50(num_classes=bottleneck)
        self.bn=nn.BatchNorm1d(bottleneck)
        self.fc1=nn.Linear(bottleneck+2,512)
        self.bn1=nn.BatchNorm1d(512)
        self.fc2=nn.Linear(512,64)
        self.bn2=nn.BatchNorm1d(64)
        self.fc3=nn.Linear(64,out_ch)
    
    def forward(self, shapes, cfd_data): 
#         # Take irfft of input shape
#         x=torch.irfft(shapes,2,signal_sizes=(224,224))
        
        # Normalize
#         x=(x-0.06854067)/0.25104755
        
        # Resize shape into 4D tensor for input
        x=shapes.view(-1, 1, 224, 224)
        
        # Go through Resnet
        x=self.resnet(x)
        x=self.bn(x)
        x=F.relu(x)
        
        # Append CFD data
        x=self.append_cfd_data(x, cfd_data)
        
        # Fully connected layer
        x=F.relu(self.bn1(self.fc1(x)))
        x=F.relu(self.bn2(self.fc2(x)))
        x=self.fc3(x)

        return x

    def append_cfd_data(self, x, cfd_data):
        # Append data
        x=torch.cat((x, cfd_data), 1)
        return x
    