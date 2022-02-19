# dataset_transformer.py 

from torchvision import transforms, datasets

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



