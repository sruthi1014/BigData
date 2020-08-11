import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision.models.resnet as resNet


class ResnetTwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Create components of a two layer neural net classifier (often
        referred to as an MLP) and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(ResnetTwoLayerNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass		    #
        #############################################################################
        self.model = resNet.resnet18(pretrained = True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self.softmax = nn.Softmax(dim = 1)
        self.upsample_image = nn.Upsample(scale_factor=7, mode = "bilinear", align_corners=True)
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the NN to
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
        # TODO: Implement the forward pass. This should take very few lines of code.#
        #############################################################################
        images = self.upsample_image(images)
        output = self.model(images.view(images.shape[0], 3, 224, 224))
        scores = self.softmax(output)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

