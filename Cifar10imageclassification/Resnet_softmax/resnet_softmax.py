import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models

class ResnetSoftmax(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Create components of a softmax classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''

        super(ResnetSoftmax, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        self.model = models.resnet18(pretrained = True)
        for param in self.model.parameters():
            param.requires_grad = False
 
        self.model.fc = nn.Linear(512,n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.upsample_images = nn.Upsample(scale_factor =7, mode ="bilinear", align_corners =True)

        
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the classifier to
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
        # TODO: Implement the forward pass. This should take very few lines of code.
        
        images = self.upsample_images(images)
        output = self.model(images.view(-1,images.shape[1],images.shape[2],images.shape[3]))
        scores = self.softmax(output)
        #############################################################################
        pass
        return scores
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################