#!/usr/bin/env bash

CONFIG=$1

python -m torch.distributed.launch --nproc_per_node=1 --master_port=43211 train.py -opt $CONFIG --launcher pytorch