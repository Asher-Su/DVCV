import torch
import torchvision.transforms as trainsforms
from matplotlib import pyplot as plt

def plotit(train_dl):
    images,labels = next(iter(train_dl))
    img = images[0]
    img_for_display = img.mul(255).byte().permute(1, 2, 0)
    plt.imshow(img_for_display)
    plt.axis('off')
    plt.title('9 client')
    plt.savefig('./9.png')