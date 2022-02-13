#!/usr/bin/env python3
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torchvision import transforms
import numpy
import sys

#numpy.set_printoptions( threshold = sys.maxsize )

img = Image.open( "./data/train/good_pear/9.jpg" )
print( "This is size of original image: ", img.size )

crop = img.resize(( 128, 128 ), Image.ANTIALIAS )
gray = ImageOps.grayscale( crop )
print( "This is size of GrayscaleCrop image: ", gray.size, "\n" )


#VIEW DATA

#plt.figure()
#plt.imshow( img, interpolation = "nearest" )
#plt.figure()
#plt.imshow( gray, interpolation = "nearest", cmap = "gray" )
#plt.show()


#Create a single neuron

#cnn = nn.Conv2d( 128, 128, 3 )
cnn = nn.Conv2d( 1, 1, 3 )                                            # create one 3/3 neuron for gray scale images
neuron_weights = cnn.weight.data.numpy()                                   #convert to numpy data for visualization
print( f"Weights in a single 3/3 neuron >>>\n {neuron_weights} " )

#Transform data to tensor
img_tensor = transforms.ToTensor()( gray ).unsqueeze_( 0 )

#Update tensor from created neuron
filtered_img_tensor = cnn( img_tensor )


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
plt.figure()
plt.imshow( filtered_img, interpolation = "nearest", cmap = "gray" )
plt.show()



