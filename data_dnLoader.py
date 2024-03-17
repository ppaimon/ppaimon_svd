import numpy as np
import torch
from torchvision import datasets, transforms

params = {
    "root": 'data', 
    "download": True, 
    "transform": transforms.Compose([
        transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,))
    ]),
}
def mnist(train):
    mnist = datasets.MNIST( train=train ,**params )
    images , labels = list(zip(*mnist))
    #images = [np.array(torch.flatten(it)) for it in images]
    images = [np.array(it[0]) for it in images]
    return np.array(images) , np.array(labels)
    