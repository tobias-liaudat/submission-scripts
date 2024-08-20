#!/bin/bash

# Load module where wandb is installed
module purge
module load pytorch-gpu/py3/2.3.0

offline_runs="$1/offline-run*"

while :
do
for ofrun in $offline_runs
do
        wandb sync $ofrun;
done
    sleep 5m
done
