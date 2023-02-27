#!/bin/bash
date 
tar -xf ../../data/cocofb.tar -C /dev/shm/
date

module load anaconda/2021.05
source activate pytorch
export PYTHONUNBUFFERED=1
torchrun --nproc_per_node=8 train.py -e=$1 -b=$2 -n=$3 -lr=$4 -ds=/dev/shm/cocofb/