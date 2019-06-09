from inception import Inception_A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class DeConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(DeConv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.relu(x)
        return x

class Minutiae1a(nn.Module):
    def __init__(self):
        super(Minutiae1a, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        
        self.block1 = nn.Sequential(
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A()
                )
        
        self.block1a = nn.Sequential(
                DeConv2d(384, 128, kernel_size=3, stride=3, padding=1),
                BasicConv2d(128, 128, kernel_size=3),
                DeConv2d(128, 32, kernel_size=3, stride=3, padding=1),
                BasicConv2d(32, 6, kernel_size=3, padding=1)
                )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block1a(x)
        return x
    
class Minutiae1b(nn.Module):
    def __init__(self):
        super(Minutiae1b, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        
        self.block1 = nn.Sequential(
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A()
                )
        
        self.block1b = nn.Sequential(
                BasicConv2d(384, 768, kernel_size=3, padding=1),
                BasicConv2d(768, 768, kernel_size=3, stride=2, padding=2),
                BasicConv2d(768, 896, kernel_size=3, padding=1),
                BasicConv2d(896, 1024, kernel_size=3, stride=2, padding=1),
                BasicConv2d(1024, 1024, kernel_size=3, padding=1),
                BasicConv2d(1024, 1024, kernel_size=3, stride=2)
                )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block1b(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return x
    
#from torchsummary import summary
#model = Minutiae1b().to(device)
#summary(model, (384, 14, 14))