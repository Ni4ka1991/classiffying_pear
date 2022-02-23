#!/usr/bin/env python3

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import torch.utils as utils

import numpy
import sys

#set all transforms on a dataset
transform_train = transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.RandomRotation( 10 ),
                                   transforms.Resize( 150 ),
                                   transforms.CenterCrop( 128 ),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])
])


dataset_train =    datasets.ImageFolder( "data/train", transform = transform_train )
dataloader_train = utils.data.DataLoader( dataset_train, batch_size = 32, shuffle = True )

model = nn.Sequential(
        nn.Conv2d( 1, 8, 5 ),
        nn.ReLU(),
        nn.MaxPool2d( 2, 2 ),

        nn.Conv2d( 8, 32, 5 ),
        nn.ReLU(),
        nn.MaxPool2d( 2, 2 ),
        
        nn.Flatten( start_dim = 1 ),

        nn.Linear( 32 * 29 * 29, 1024 ),
        nn.ReLU(),

        nn.Linear( 1024, 256 ),
        nn.ReLU(),

        nn.Linear( 256, 2 ),

        )

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam( model.parameters(), lr = 0.001, weight_decay = 0.0001 )

print( model )
"""
for epoch in range( 10 ):
    running_loss = 0.0
    for i, data in enumerate( dataloader_train, 0 ):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model( inputs )
        loss = criterion( outputs, labels )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

if epoch % 1 == 0:
    print( "Epoch: {} \t Training loss: {:.6f}".format( epoch, running_loss ) )
"""









