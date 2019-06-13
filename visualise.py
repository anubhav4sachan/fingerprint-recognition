from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.misc import imsave

def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def save(img, name, folder):
    npimg = img.detach().numpy()
    x = np.transpose(npimg, (1, 2, 0))
    imsave("./" + str(folder) + "/1/" + name + ".jpg", x)
    
def plot_pores():   
    filelist = []
    for file in os.listdir('pore_cord_train'):
        filelist.append("pore_cord_train\\" + "\\" + file)        
    df = []    
    for name in filelist:
       df.append(pd.read_table(name, delim_whitespace=True, names=('X', 'Y')))
       
    fin = pd.concat(df)        
    fin.plot(x = 'X', y = 'Y', kind = 'scatter')