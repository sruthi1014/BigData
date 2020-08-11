 #!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 300 \
    --epochs 20\
    --weight-decay 0.00025 \
    --momentum 0.9 \
    --batch-size 128 \
    --optimizer sgd\
    --lr 0.001 | tee mymodel_new.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################