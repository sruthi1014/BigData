#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model resnet_mymodel \
    --kernel-size 1 \
    --hidden-dim 120 \
    --epochs 1\
    --weight-decay 0.0 \
    --momentum 0.9 \
    --batch-size 512 \
    --optimizer adagrad\
    --lr 0.005 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
