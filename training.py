from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from stn import STNet, stn_train, stn_test
from stem import Stem
from visualise import imshow, save, plot_pores
from texture import Texture
from minutiae import Minutiae1a, Minutiae1b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextureNet(nn.Module):
    def __init__(self):
        super(TextureNet, self).__init__()
        
        self.fc = nn.Linear(256, 1)
        self.feature1 = nn.Sequential(
                STNet(),
                Stem(),
                Texture()
                )
        self.inp = 0
        
    def forward(self, x):
        x11 = self.feature1(x)
        x1 = x11.view(-1, 256)
        x1 = self.fc(x1) 
        self.inp = x1
        return x1, x11

class Minutiae2aNet(nn.Module):
    def __init__(self):
        super(Minutiae2aNet, self).__init__()
        self.feature2a = nn.Sequential(
                STNet(),
                Stem(),
                Minutiae1a()
                )
        
    def forward(self, x):
        x2a = self.feature2a(x)
        return x2a
    
class Minutiae2bNet(nn.Module):
    def __init__(self):
        super(Minutiae2bNet, self).__init__()
        self.fc = nn.Linear(256, 1)
        self.feature2b = nn.Sequential(
                STNet(),
                Stem(),
                Minutiae1b()
                )
        self.inp = 0
        
    def forward(self, x):
        x2b2 = self.feature2b(x)
        x2b = x2b2.view(-1, 256)
        x2b = self.fc(x2b)
        self.inp = x2b
        return x2b, x2b2   

model1 = TextureNet().to(device)
model2a = Minutiae2aNet().to(device)
model2b = Minutiae2bNet().to(device)

criterion1 = nn.CrossEntropyLoss()
criterion2a = nn.MSELoss()
criterion2b = nn.CrossEntropyLoss()

optimizer1 = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
optimizer2a = optim.SGD(model2a.parameters(), lr=0.001, momentum=0.9)
optimizer2b = optim.SGD(model2b.parameters(), lr=0.001, momentum=0.9)

Transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                            torchvision.transforms.Resize((132,132)),
                                            torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(root='./imgs/', 
                                           transform=Transform)        
#test_dataset = torchvision.datasets.ImageFolder(root='./imgs/test/',
#                                                transform=Transform)

train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
#test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

#train_loader_org=torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
#
#stn_test(train_loader, 'shuffled')
#
#trainloader=torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root='./shuffled/', 
#                                          transform=Transform), batch_size=1, shuffle=True)

#
#trainloader_org=torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root='./aligned/original', 
#                                          transform=Transform), batch_size=1, shuffle=False)


# Train the model

#for j in range (10):
#    model1.train()
#    model2a.train()
#    model2b.train()
#    
#    for i, data in enumerate(trainloader):
#        inputs, target = data[0].to(device), data[1].to(device)
#        target_f = torch.Tensor.float(target)
#        
#        optimizer1.zero_grad()
#        optimizer2a.zero_grad()
#        optimizer2b.zero_grad()
#        
#        output1 = model1(inputs)[0]
#        output2a = model2a(inputs)
#        output2b = model2b(inputs)[0]
#        
#        loss = (criterion1(F.softmax(output1), target) + criterion2a(output2a, target_f) + 
#                criterion2b(F.softmax(output2b), target))
#        loss.backward()
#        
#        optimizer1.step()
#        optimizer2a.step()
#        optimizer2b.step()
#        
#        print(j, i, loss.item())
        
#torch.save(model1, 'trained1.pth')
#torch.save(model2a, 'trained2a.pth')
#torch.save(model2b, 'trained2b.pth')
