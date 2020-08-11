import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.conv1 = torch.nn.Conv2d(im_size[0], hidden_dim*3, kernel_size,padding=1,stride=1)
        self.pool = torch.nn.MaxPool2d(3, 1)
        self.conv2 = torch.nn.Conv2d(hidden_dim*3, hidden_dim*2, kernel_size+2,padding=1,stride=1)
        self.conv3 = torch.nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size,padding=1,stride=1)
        self.conv4 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size+2,1,1)
        self.fc1 = torch.nn.Linear(hidden_dim *20 * 20, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, n_classes)
       # self.bn2=torch.nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True)
        self.Softmax = torch.nn.Softmax(dim=1)
        self.dropout=torch.nn.Dropout(0.50)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
       # print(images.size())
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        output = self.pool(self.conv1(images))
        x=functional.relu(output)
        #print(x.size())
        x=functional.relu(self.pool(self.dropout(self.conv2(x))))
        #print(x.size())
        x=functional.relu(self.pool(self.conv3(x)))
        #print(x.size())
        x=functional.relu(self.pool(self.dropout(self.conv4(x))))
       #x=self.pool(x)
        #print(x.size())
        x = x.view(x.shape[0], -1)
        #print(x.size())
        x = functional.relu(self.fc1(x))
        x=self.fc2(x)
        scores = self.Softmax(x)
       # print(scores)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores