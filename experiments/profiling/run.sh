#!/bin/bash

export TMPDIR=/state/partition1/user/$USER
mkdir -p $TMPDIR

eval "$(conda shell.bash hook)"

conda activate synthax
echo "Running synthax profiling experiments on GPU"
CUDA_VISIBLE_DEVICES=0 python3 -u profile_synthax.py
# echo "Running synthax profiling experiments on CPU"
# CUDA_VISIBLE_DEVICES="-1" python3 -u profile_synthax.py

conda activate esc
echo "Running torchsynth profiling experiments on GPU"
CUDA_VISIBLE_DEVICES=0 python3 -u profile_torchsynth.py
# echo "Running torchsynth profiling experiments on CPU"
# CUDA_VISIBLE_DEVICES="-1" python3 -u profile_torchsynth.py
