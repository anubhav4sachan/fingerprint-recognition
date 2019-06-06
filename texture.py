from inception import Inception_A, Inception_B, mInception_B, Inception_C, mInception_C, BasicConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Texture(nn.Module):
    def __init__(self):
        super(Texture, self).__init__()
#        self.fc = nn.Linear(1536 * 14 * 14, 256)
        
        self.block1 = nn.Sequential(
                Inception_A(),
                Inception_A(),
                Inception_A(),
                Inception_A(),
                mInception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                Inception_B(),
                mInception_C(),
                Inception_C(),
                Inception_C()
                )
        
    def forward(self, x):
        out = self.block1(x)
#        out = out.view(-1, 14*14*1536)
#        out = self.fc(out)
        return out

model = Texture().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

loss_type = nn.MSELoss() 
      
def texture_train(epoch, train_loader):
    for e in range(epoch):
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = torch.Tensor.float(labels)
            
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_type(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print(e, i, loss.item())
            
#from torchsummary import summary
#model = Texture().to(device)
#summary(model, (384, 14, 14))
        
