import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        
        #nn.Conv2d(x, y, z)
        #x = No. of Input Channels i.e. 3 in our case R,G,B
        #y = No. of Channels to be produced by Convolution Layer
        #    i.e. No. of Filters to be used
        #z = Filter Size i.e. 5 * 5 in our case
        
        #global filter_size
        self.filter_size = kernel_size
        self.im_size     = im_size 
        op_size   = im_size[1] - kernel_size + 1  #Output Image Size = N - F + 1
        
        self.conv1 = nn.Conv2d(im_size[0], hidden_dim, kernel_size)
        #Maxpool2d is used to reduce the High dimensionality of the image.
        self.pool =  nn.MaxPool2d(2, 2)
 
        
        pool_x    = op_size/2
        pool_y    = op_size/2

        #The Fully Connected Layers are connected to the output layer
        self.fc1   = nn.Linear(6 * 14 * 14, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

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
  
        images = self.pool(functional.relu(self.conv1(images)))
        #images = self.pool(functional.relu(self.conv2(images)))
        
        filter_size = self.filter_size  #Filter Size
        im_size     = self.im_size
        no_filters  = 6

        op_size   = im_size[1] - filter_size + 1  #Output Image Size = N - F + 1
        pool_x    = op_size/2
        pool_y    = op_size/2
        print("images size", images.size())
        images = images.view(-1, 6 * 14 * 14)
        images = functional.relu(self.fc1(images))
        print("size - ", images.size())
        images = functional.relu(self.fc2(images))
        scores = functional.relu(self.fc3(images))

        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

