from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()