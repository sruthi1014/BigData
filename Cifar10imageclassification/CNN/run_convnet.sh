#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 5 \
    --hidden-dim 6 \
    --epochs 1 \
    --weight-decay 0.0 \
    --momentum 0.1 \
    --batch-size 128 \
    --optimizer sgd \
    --lr 0.005 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
