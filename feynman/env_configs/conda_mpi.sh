#!/bin/bash
module purge
module load miniforge/24.7.1
module intel/2021.4.0
module load openmpi4/4.1.1

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

