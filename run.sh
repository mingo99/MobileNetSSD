#!/bin/bash
module load anaconda/2021.05
source activate pytorch
export PYTHONUNBUFFERED=1
python train.py -e=$1 -b=$2 -lr=$3