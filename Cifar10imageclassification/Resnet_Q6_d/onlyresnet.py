import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models.resnet import resnet18 as resnet18
from torchvision import transforms


class ONLYRESNET(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
    
        super(ONLYRESNET, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        
        import torchvision.models as models
        
        #Begin of New Code
        
        self.resnet18_full        = resnet18(pretrained=True)
        
        #self.resnet18 = nn.Sequential(*list(self.resnet18_full.children())[:-1])
        
        #for param in self.resnet18.parameters():
        #        param.requires_grad= False

        #self.upsample_img    = nn.Upsample(scale_factor=7, mode="bilinear", align_corners=True)
        #End of New Code
               
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #print("Forward Pass Entered")
        #images = self.upsample_img(images)
        scores = self.resnet18_full(images)
       
        #a = images.size(0)
        #b = images.size(1)/32
        #c = 8 
        #b = int(b)
        #d = 4
        
        #if (b == 16): #For the Rest Batches
        #  images = images.reshape(a, 8, 8, 8)
        #  images = self.pool(self.conv2D(images))
        #if (b != 16):  #For the Last Batch
        #  images = images.reshape(a, 8, 13, 4)
        #  images = self.pool(self.conv2D(images))
        # 
        #images = images.view(-1, images.size(1) * images.size(2) * images.size(3))

        #scores = functional.relu(self.fc1(images))

        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

