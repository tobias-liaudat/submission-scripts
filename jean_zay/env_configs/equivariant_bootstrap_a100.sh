#!/bin/bash
# Environment for the equivariant-bootstrap UQ campaign, A100 partition.
# The arch/a100 pre-module MUST be loaded before pytorch-gpu.
module purge
module load arch/a100
module load pytorch-gpu/py3/2.7.0
