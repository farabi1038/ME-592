
import os
import numpy as np
import pandas as pd
from PIL import Image
from time import time
from matplotlib import pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary



model = torch.load('/Users/ibnefarabishihab/Desktop/Course Materials/ME 592/task2_50.pth')

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((300,300)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20),
    torchvision.transforms.ToTensor()
])

img = Image.open('/Users/ibnefarabishihab/Desktop/Course Materials/ME 592/hw3/cropped/')
img_tensor = preprocess(img)
img_tensor.unsqueeze_(0)
output = model(Variable(img_tensor))