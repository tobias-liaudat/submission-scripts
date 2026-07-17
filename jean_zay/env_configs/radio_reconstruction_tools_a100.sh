#!/bin/bash
# Environment for radio_reconstruction_tools on the Jean Zay A100 partition (gpu_p5).
# The arch/a100 pre-module MUST be loaded before any other module to get the
# A100-compatible builds.
module purge
module load arch/a100
module load pytorch-gpu/py3/2.7.0
