#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model resnet_softmax \
    --hidden-dim 200 \
    --epochs 2 \
    --weight-decay 0.0001 \
    --momentum 0.1 \
    --batch-size 128 \
    --lr 0.01 | tee resnet_twolayernn-a.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
