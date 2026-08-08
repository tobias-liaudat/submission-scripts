#!/bin/bash
# Environment for the equivariant-bootstrap UQ campaign, H100 partition.
# The arch/h100 pre-module MUST be loaded before pytorch-gpu.
module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0
