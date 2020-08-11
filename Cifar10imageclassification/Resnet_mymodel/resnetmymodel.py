import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models

class ResnetMyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(ResnetMyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.model1= models.resnet18(pretrained=True)
        print("check 7")
        #print(self.model1.fc)
        num_ftrs = self.model1.fc.in_features
        #print(num_ftrs)
        del self.model1.fc
        #print(self.model.fc)
        self.model1.beforefc = torch.nn.Sequential(
           torch.nn.Conv2d(num_ftrs,512,kernel_size,padding=1),
           torch.nn.MaxPool2d(2, 2),
           torch.nn.Conv2d(512, 128, kernel_size,padding=1),
           torch.nn.Conv2d(128, 128, kernel_size),
           torch.nn.MaxPool2d(2, 2),
        )
        self.model1.fc=torch.nn.Linear(128 * 1 * 1, n_classes)
        self.resize=torch.nn.Upsample(scale_factor=2,mode='bilinear')
       # print(self.model1)
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
        #print(images)
        y= self.resize(images)
        #print(y.size(), images[0])
        y=y.reshape(y.shape[0],64,64,3)
        #print(images.size(),y.size())
        # print(scores)
        #print(self.model.fc)
        #print("check 8")
        output = self.model1.layer1(y)
       # print(" layer 1 done", output.size())
        output=self.model1.layer2(output)
        #print(" layer 2 done",output.size())
        output=self.model1.layer3(output)
        #print(" layer 3 done", output.size())
        output=self.model1.layer4(output)
        #print(" layer 4 done", output.size())
        output=self.model1.avgpool(output)
        #print(" layer 5 done", output.size())
        #print(output)
        output=self.model1.beforefc(output)
        #print(" layer before fc done", output.size())
        output = output.view(-1, 128 * 1 * 1)
        #flatten the output
        scores=self.model1.fc(output)
        scores = functional.softmax(scores,dim=1)
        #print(" layer 6 done", scores.size())
        #print(scores)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

