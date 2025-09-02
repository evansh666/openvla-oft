#!/bin/bash
eval "$(conda shell.bash hook)"

# Create and activate conda environment
conda deactivate
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# install openvla-oft environment
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# install for RLDS dataset: tensorflow, tensorflow_datasets, tensorflow_hub, apache_beam
pip install tensorflow_hub
pip install apache_beam


# install behavior evaluation environment
cd ../BEHAVIOR-1K # go to BEHAVIOR-1K directory
# install bddl
cd bddl
pip install -e .
pip install pymeshlab==2022.2.post4

# install omnigibson with eval dependencies
cd ..
pip install Omnigibson[eval]
