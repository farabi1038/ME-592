import os
import numpy as np
import pandas as pd
from PIL import Image
from time import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

dataset_path = '/home/reuel/Desktop/Sakib/hw3/Leaf_Images/'

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


# Transformation function to be applied on images
# 1. Horizontally Flip the image with a probability of 30%
# 2. Randomly Rotate the image at an angle between -40 to 40 degress.
# 3. Resize each images to a smallest size of 300 pixels maintaining aspect ratio
# 4. Crop a square of size 256x256 from the center of image
# 5. Convert Image to a Pytorch Tensor
# 6. Normalize the pytorch's tensor using mean & std of imagenet


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((400,400)),
    torchvision.transforms.ToTensor()
])



batch_size=64
num_workers=4


dataset = torchvision.datasets.ImageFolder(root = dataset_path, transform=transforms)
train, test = torch.utils.data.random_split(dataset, [11781, 2944])
train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                           num_workers=num_workers, drop_last=True, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                          num_workers=num_workers, drop_last=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(),
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
            nn.Linear(16384, 256),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(256, 10)
        ).to(device)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


model = MyModel().to(device)
summary(model, (3, 400, 400))


def Train(epoch, print_every=50):
    total_loss = 0
    start_time = time()

    accuracy = []

    for i, batch in enumerate(train_dataloader, 1):
        minput = batch[0].to(device)  # Get batch of images from our train dataloader
        target = batch[1].to(device)  # Get the corresponding target(0, 1 or 2) representing cats, dogs or pandas

        moutput = model(minput)  # output by our model

        loss = criterion(moutput, target)  # compute cross entropy loss
        total_loss += loss.item()

        optimizer.zero_grad()  # Clear the gradients if exists. (Gradients are used for back-propogation.)
        loss.backward()  # Back propogate the losses
        optimizer.step()  # Update Model parameters

        argmax = moutput.argmax(dim=1)  # Get the class index with maximum probability predicted by the model
        accuracy.append(
            (target == argmax).sum().item() / target.shape[0])  # calculate accuracy by comparing to target tensor

        if i % print_every == 0:
            print('Epoch: [{}]/({}/{}), Train Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
                epoch, i, len(train_dataloader), loss.item(), sum(accuracy) / len(accuracy), time() - start_time
            ))

    return total_loss / len(train_dataloader)  # Returning Average Training Loss


def Test(epoch):
    total_loss = 0
    start_time = time()

    accuracy = []

    with torch.no_grad():  # disable calculations of gradients for all pytorch operations inside the block
        for i, batch in enumerate(test_dataloader):
            minput = batch[0].to(device)  # Get batch of images from our test dataloader
            target = batch[1].to(device)  # Get the corresponding target(0, 1 or 2) representing cats, dogs or pandas
            moutput = model(minput)  # output by our model

            loss = criterion(moutput, target)  # compute cross entropy loss
            total_loss += loss.item()

            # To get the probabilities for different classes we need to apply a softmax operation on moutput
            argmax = moutput.argmax(
                dim=1)  # Find the index(0, 1 or 2) with maximum score (which denotes class with maximum probability)
            accuracy.append((target == argmax).sum().item() / target.shape[
                0])  # Find the accuracy of the batch by comparing it with actual targets

    print('Epoch: [{}], Test Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
        epoch, total_loss / len(test_dataloader), sum(accuracy) / len(accuracy), time() - start_time
    ))
    return total_loss / len(test_dataloader)  # Returning Average Testing Loss




lr = 0.0001
model = MyModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

Test(0)

train_loss = []
test_loss = []

for epoch in range(1, 51):
    train_loss.append(Train(epoch, 200))
    test_loss.append(Test(epoch))

    print('\n')

    if epoch % 10 == 0:
        torch.save(model, 'modelTask1p2_' + str(epoch) + '.pth')

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_loss)+1), train_loss, 'g', label='Training Loss')
plt.plot(range(1, len(test_loss)+1), test_loss, 'b', label='Testing Loss')

plt.title('Training and Testing loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('learning curve.png')
