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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.model = nn.Sequential(
            #2,3,300,300
            nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(),
            #
            nn.Conv2d(16, 16, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),

        ).to(device)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(1024, 256),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(256, 10)
        ).to(device)

    def forward(self, x):
        #print('enter',x.shape)
        x = self.model(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x




model = MyModel()

model = torch.load('/Users/ibnefarabishihab/Desktop/Course Materials/ME 592/task2_50.pth',map_location=torch.device('cpu'))


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256,256)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20),
    torchvision.transforms.ToTensor()
])

path='/Users/ibnefarabishihab/Desktop/Course Materials/ME 592/hw3/cropped/'
for i in os.listdir(path):
    img = Image.open(path+str(i))
    img_tensor = transforms(img)
    img_tensor.unsqueeze_(0)
    output = model(img_tensor)
    argmax = output.argmax(
                dim=1)
    print('name of image :'+str(i),'prediction: '+str(int(argmax)+1))