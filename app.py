#!/usr/bin/env python3

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torchvision import transforms, datasets
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


dataset_train = datasets.ImageFolder( "data/train", transform = transform_train )
dataloader_train = torch.utils.data.DataLoader( dataset_train, batch_size = 32, shuffle = True )



















"""
#Create a single neuron

#cnn = nn.Conv2d( 1, 8, 5 )                                                 # create one 5/5 neuron for gray scale images and eight(8) outputs
#neuron_weights = cnn.weight.data.numpy()                                   # convert to numpy data for visualization
#print( f"Weights in a single 5/5 neuron >>>\n {neuron_weights} " )

#Transform data to tensor
#img_tensor = transforms.ToTensor()( gray ).unsqueeze_( 0 )

#Update tensor from created neuron
#filtered_img_tensor = cnn( img_tensor )



#View tensors and his properties (shape)

print( )
print( f"Image to tensor print>>>\n {img_tensor}" )
print( "The end of image tensor" )
print( "*"*22 )
print( f"Image tensor SHAPE: {img_tensor.shape}" )
print( "*"*22 )
print()
print( f"Filtered img tensor SHAPE: {filtered_img_tensor.shape}" )
print( "*"*22 )
print( f" Filtered img tensor >>>\n {filtered_img_tensor}" )
print( "The end" )

#transform filtered tensor in img
filtered_img = transforms.ToPILImage()( filtered_img_tensor.squeeze_( 0 ))

#view filtered img  
#plt.figure()
#plt.imshow( filtered_img, interpolation = "nearest", cmap = "gray" )
#plt.show()
"""


