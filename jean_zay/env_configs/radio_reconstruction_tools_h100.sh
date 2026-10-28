#!/bin/bash
# Environment for radio_reconstruction_tools on the Jean Zay H100 partition.
#
# Identical to radio_reconstruction_tools.sh except for `module load arch/h100`,
# which must come FIRST. Without it the default (a100/v100) build of
# pytorch-gpu/py3/2.7.0 is selected, and it fails at `import torch` on an H100
# node with:
#
#     OSError: libmpi.so.40: cannot open shared object file
#
# so this is a missing-module error rather than anything to do with the code.
# Kept as a separate file instead of branching inside the shared one: the job
# already has to choose `-C h100 -A <acct>@h100 --qos=qos_gpu_h100-*`, and a
# script that silently picked an arch from the node it happened to land on
# would be harder to reason about than one the job names explicitly.
module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0
