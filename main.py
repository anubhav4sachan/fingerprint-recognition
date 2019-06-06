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
from visualise import imshow
from texture import Texture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                            torchvision.transforms.Resize((132,132)),
                                            torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(root='./imgs/train/', 
                                           transform=Transform)  
      
test_dataset = torchvision.datasets.ImageFolder(root='./imgs/test/',
                                                transform=Transform)

train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)   

    
stn_train(8, train_loader)

pred_r = stn_test(test_loader)

print("original test images")   
dataiter = iter(test_loader)
for i in range (1):
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

print("predicted test images")
imshow(torchvision.utils.make_grid(pred_r[0].cpu()))

o = []
o.append(Stem().forward(pred_r[0].cpu()))

out = Texture().forward(o[0])
#o = torch.stack(o) #list -> tensor conversion
for batch_idx, batch in enumerate(out):
    print(batch[0].size())
    imshow(torchvision.utils.make_grid(batch[0]))