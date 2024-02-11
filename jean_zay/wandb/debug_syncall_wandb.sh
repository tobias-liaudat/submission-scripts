#!/bin/bash

# Load module where wandb is installed
module purge
module load pytorch-gpu/py3/2.1.1

offline_runs="$1/offline-run*"

while :
do
for ofrun in $offline_runs
do
        echo $ofrun;
done
    sleep 1s
done
