from inception import Inception_A, Inception_B, mInception_B, Inception_C, mInception_C, BasicConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Texture(nn.Module):
    def __init__(self):
        super(Texture, self).__init__()
        self.fc = nn.Linear(1536 * 14 * 14, 256)
        
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
        out = out.view(-1, 14*14*1536)
        out = self.fc(out)
        return out
            
#from torchsummary import summary
#model = Texture().to(device)
#summary(model, (384, 14, 14))