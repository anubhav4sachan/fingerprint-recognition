import torch
import torch.nn as nn

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

class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        
        self.layer1 = nn.Sequential(
                BasicConv2d(1, 3, kernel_size=5, padding=2),
                BasicConv2d(3, 32, kernel_size=3, stride=2),
                BasicConv2d(32, 32, kernel_size=3),
                BasicConv2d(32, 64, kernel_size=3, padding=1)
                )
        self.layer1A = nn.MaxPool2d(3, stride=2)
        self.layer1B = BasicConv2d(64, 96, kernel_size=3, stride=2)
        
        self.layer2A = nn.Sequential(
                BasicConv2d(160, 64, kernel_size=1),
                BasicConv2d(64, 96, kernel_size=3)
                )
        self.layer2B = nn.Sequential(
                BasicConv2d(160, 64, kernel_size=1),
                BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
                BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
                BasicConv2d(64, 96, kernel_size=3)
                )
        
        self.layer3A = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.layer3B = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        lay1 = self.layer1(x)
        lay1A = self.layer1A(lay1)
        lay1B = self.layer1B(lay1)
        
        lay2 = torch.cat((lay1A, lay1B), dim=1)
        lay2A = self.layer2A(lay2)
        lay2B = self.layer2B(lay2)
        
        lay3 = torch.cat((lay2A, lay2B), dim=1)
        lay3A = self.layer3A(lay3)
        lay3B = self.layer3B(lay3)
        
        out = torch.cat((lay3A, lay3B), dim=1)
        
        return out
    
#from torchsummary import summary
#model = Stem().to(device)
#summary(model, (1, 132, 132))