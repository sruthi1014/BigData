#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model resnet_twolayernn \
    --hidden-dim 675 \
    --epochs 40 \
    --optimizer sgd \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --batch-size 256 \
    --lr 0.001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
