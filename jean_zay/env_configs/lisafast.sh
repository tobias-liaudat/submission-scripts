#!/bin/bash
module purge # disable the external environment

# Load all the modules
module load anaconda-py3/2023.09 cuda/12.2.0 cudnn/9.8.0.87-cuda nccl/2.19.3-1-cuda

# Activate your Python virtual environment
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
source activate jax-0.6.0+py3.11.5
