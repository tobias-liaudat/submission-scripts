#!/bin/bash
#SBATCH --job-name=test_radio_sim
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10    #ask for 10 cpus
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
##SBATCH --mem-per-cpu=1G
#SBATCH --output=-%x_%A_%a.log

# Load the required modules
module load miniforge/24.7.1
module intel/2021.4.0
module load openmpi4/4.1.1

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
source activate karabo_env

set -x

# go to script dir
cd /feynman/home/dedip/lilas/tl255879/work/projects/Numpex/radio_simulations/repos/Karabo-Pipeline/karabo/PnP

python -c "import torch;print(torch.cuda.is_available());print(torch.get_num_threads());torch.set_num_threads(max(1, torch.get_num_threads() // 2))"
nvidia-smi

# Run script
python simulation_script.py

# Return exit code
exit 0