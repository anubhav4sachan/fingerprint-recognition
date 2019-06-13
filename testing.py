from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = torch.load('./trained1.pth')
model1 =  model1.eval()

model2a = torch.load('trained2a.pth')
model2a =  model2a.eval()

model2b = torch.load('trained2b.pth')
model2b =  model2b.eval()

Transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                            torchvision.transforms.Resize((132,132)),
                                            torchvision.transforms.ToTensor()])

dataset1 = torchvision.datasets.ImageFolder(root='./test_imgs/set1/', 
                                           transform=Transform)

dataset2 = torchvision.datasets.ImageFolder(root='./test_imgs/set2/', 
                                           transform=Transform)

loader1=torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=False)
loader2=torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False)

t1= []      #T_test 
for data in loader1:
    img = data[0].to(device)
    output1 = model1(img)[1]
    output2 = model2b(img)[1]
    t1.append(torch.cat((output1, output2), dim=1))
    
t2 = []       #T_original
for data in loader2:
    img = data[0].to(device)
    o1 = model1(img)[1]
    o2 = model2b(img)[1]
    t2.append(torch.cat((o1, o2), dim=1))
    
#Calculating similarity scores

scores = []
for i in t1:
    score = []
    for j in t2:
        score.append(F.cosine_similarity(i, j, dim=1).item())
    scores.append(score)
    
one_gen = []
two_gen = []
f = 0
for k in scores:
    if f < 4:
        one_gen.append(k[:4])
        f += 1
    if f >= 4 and f < 8:
        two_gen.append(k[4:])
        f += 1

one_imp = []
two_imp = []
f= 0
for l in scores:
    if f < 4:
        one_imp.append(l[4:])
        f += 1
    if f >= 4 and f < 8:
        two_imp.append(l[:4])
        f += 1

print(two_gen)
print('')
print(two_imp)