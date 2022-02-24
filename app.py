#!/usr/bin/env python3

from dataset_transformer import *
from nn_configuration import *

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch, torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import numpy
import sys

#apply transforms regulations
dataset_train =    datasets.ImageFolder( "data/train", transform = transform_train )

dataloader_train = utils.data.DataLoader( dataset_train, batch_size = 2, shuffle = True )

#print( model )

#print( dataloader_train )
print( len( dataloader_train ))

input( "hit enter ..." )


#for epoch in range( 3 ):
#    running_loss = 0.0

#    for i, data in enumerate( dataloader_train, 0 ):

#        print( "i    >>>> {}".format( i ))
#        print( "data >>>> {}".format( data ))


#        inputs, labels = data
#        optimizer.zero_grad()
#        outputs = model( inputs )
#        loss = criterion( outputs, labels )
#        loss.backward()
#        optimizer.step()
#        running_loss += loss.item()

#if epoch % 1 == 0:
#    print( "Epoch: {} \t Training loss: {:.6f}".format( epoch, running_loss ) )








