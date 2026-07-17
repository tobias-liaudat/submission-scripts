#!/bin/bash
# Environment for radio_reconstruction_tools on the Jean Zay H100 partition (gpu_p6).
# The arch/h100 pre-module MUST be loaded before any other module to get the
# H100-compatible builds.
module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0
