#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model resnet \
    --epochs 5 \
    --weight-decay 0.001 \
    --momentum 0.9 \
    --batch-size 128 \
    --optimizer sgd \
    --lr 0.001 | tee resnet_softmax_a.log