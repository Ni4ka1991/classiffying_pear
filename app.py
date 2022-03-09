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
dataset_train =    datasets.ImageFolder( "sml_data/train", transform = transform_train )
#img = Image.open( "sml_data/train/good_pear/pera.jpg" )

plt.figure()
plt.imshow( dataset_train, interpolation = "nearest" )
#plt.figure()
#plt.imshow( gray, interpolation = "nearest", cmap = "gray" )
plt.show()



#dataloader_train = utils.data.DataLoader( dataset_train, batch_size = 1, shuffle = True )

#print( model )


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








