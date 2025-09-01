#!/bin/bash
eval "$(conda shell.bash hook)"

# Create and activate conda environment
conda deactivate
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

mkdir -p checkpoints
mkdir -p datasets

# install for RLDS dataset: tensorflow, tensorflow_datasets, tensorflow_hub, apache_beam
pip install tensorflow_hub
pip install apache_beam

cd ../BEHAVIOR-1K
./setup.sh --omnigibson --bddl --eval