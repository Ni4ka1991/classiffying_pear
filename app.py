#!/usr/bin/env python3

from dataset_transformer import *
from nn_configuration import *

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch, torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import numpy as np
import sys

#apply transforms regulations
dataset_train =    datasets.ImageFolder( "data/train", transform = transform_train )
#print( f"type( dataset_train ) >>> {type( dataset_train )}" )
#print( f"Dataset_len ( data/train ) >>> {len( dataset_train )}" )
#print(f"\ndataset_getitem[15] >>>\n{dataset_train[15]}")
#print(f"\ndataset_getitem[16] >>>\n{dataset_train[16]}")
#print(f"\ndataset_getitem[16][0] >>>\n{dataset_train[16][0]}")
#print(f"\ndataset_getitem[16][1] >>>\n{dataset_train[16][1]}")
#dataset_train_16 = np.array( dataset_train[16][0] )
#print(f"\nShape of dataset_getitem[16][0] >>>  {dataset_train_16.shape}")
#input( "hit enter ..." )

dataloader_train = utils.data.DataLoader( dataset_train, batch_size = 1, shuffle = True )

#print( model )

#print( dataloader_train )
#print( len( dataloader_train ))
#print( f"\nLen of dataloader_train >>>{len( dataloader_train )}" )
#print( f"\ntype( dataloader_train )  >>>{type( dataloader_train )}" )

for epoch in range( 3 ):
    running_loss = 0.0
    
    for i, data in enumerate( dataloader_train, 0 ):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model( inputs )
        loss = criterion( outputs, labels )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

if epoch %1 == 0:
    print( "Epoch: {} \tTraining Loss: {:.6f}".format( epoch, running_loss ))








