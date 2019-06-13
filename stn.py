from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from visualise import save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Viewing size of a tensor.
#for batch_idx, batch in enumerate(train_loader):
#    print(batch_idx, batch[0].size())

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
   
class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()

        self.localization = nn.Sequential(
            BasicConv2d(1, 24, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            BasicConv2d(24, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            BasicConv2d(48, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 3 * 2),
        )

        self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 64 * 8 * 8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        return x


model = STNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0001)

loss_type = nn.MSELoss() 
      
def stn_train(epoch, train_loader):
    for e in range(epoch):
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = torch.Tensor.float(labels)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_type(outputs, labels)
            loss.backward()
            optimizer.step()
            print(e, i, loss.item())
             
#print("original images")   
#dataiter = iter(test_loader)
#images, labels = dataiter.next() #Use for loops to print more
#imshow(torchvision.utils.make_grid(images))


# 'stn_test' is the output of the STNet
            
def stn_test(test_loader, folder):
    i = 0
    with torch.no_grad():
        for data in test_loader:
            images = data[0].to(device)
            output = model(images)
            i += 1
            save(torchvision.utils.make_grid(output.cpu()), str(i), folder)
        
        
#pred_r = test()
#print("predicted images")
#imshow(torchvision.utils.make_grid(pred_r[0].cpu()))
    
#from torchsummary import summary
#summary(model, (1, 132, 132))