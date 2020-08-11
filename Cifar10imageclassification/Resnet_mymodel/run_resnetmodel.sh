#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model resnet_mymodel \
    --kernel-size 1 \
    --hidden-dim 120 \
    --epochs 5\
    --weight-decay 0.0 \
    --momentum 0.9 \
    --batch-size 256 \
    --optimizer adagrad\
    --lr 0.005 | tee resnetmymodel_b.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
