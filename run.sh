#!/bin/bash
date 
tar -xf ../../data/cocofb.tar -C /dev/shm/
date

module load anaconda/2021.05
source activate pytorch
export PYTHONUNBUFFERED=1
torchrun --nproc_per_node=$1 train.py -e=$2 -b=$3 -n=$4 -lr=$5 --resume=$6 -ds=/dev/shm/cocofb/ -y=2023 --eval_freq=5