#!/bin/bash
#SBATCH --job-name=PR_original_dataset_benchmark_wfv1_test_n1_    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=PR_original_dataset_benchmark_wfv1_test_n1_%j.out  # nom du fichier de sortie
#SBATCH --error=PR_original_dataset_benchmark_wfv1_test_n1_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-use=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A ynx@v100                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --array=0-3

# Load env
cd /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/phase_retrieval_paper/repo/submission-scripts/jean_zay/env_configs/
. wfv1.sh

# echo des commandes lancees
set -x

opt[0]="--id_name _original_dataset_wf_PR_benchmark_2_cycles_v0 --random_seed 5000"
opt[1]="--id_name _original_dataset_wf_PR_benchmark_2_cycles_v1 --random_seed 5100"
opt[2]="--id_name _original_dataset_wf_PR_benchmark_2_cycles_v2 --random_seed 5200"
opt[3]="--id_name _original_dataset_wf_PR_benchmark_2_cycles_v3 --random_seed 5300"

cd /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/phase_retrieval_paper/repo/wf-psf/long-runs/

srun python -u ./train_eval_plot_script_click_multi_cycle.py \
    --base_id_name _original_dataset_wf_PR_benchmark_2_cycles_ \
    --total_cycles 2 \
    --saved_cycle cycle2 \
    --eval_only_param False \
    --reset_dd_features False \
    --eval_only_param False \
    --project_dd_features False \
    --d_max 2 \
    --n_zernikes 45 \
    --save_all_cycles True \
    --n_bins_lda 20 \
    --n_bins_gt 20 \
    --opt_stars_rel_pix_rmse True \
    --pupil_diameter 256 \
    --n_epochs_param_multi_cycle "15 15" \
    --n_epochs_non_param_multi_cycle "100 50" \
    --l_rate_non_param_multi_cycle "0.1 0.06" \
    --l_rate_param_multi_cycle "0.01 0.004" \
    --model poly \
    --model_eval poly \
    --cycle_def complete \
    --gt_n_zernikes 45 \
    --d_max_nonparam 5 \
    --saved_model_type checkpoint \
    --use_sample_weights True \
    --l2_param 0. \
    --interpolation_type none \
    --eval_batch_size 16 \
    --train_opt True \
    --eval_opt True \
    --plot_opt True \
    --dataset_folder /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/phase_retrieval_paper/repo/wf-psf/data/coherent_euclid_dataset/ \
    --test_dataset_file test_Euclid_res_id_001.npy \
    --train_dataset_file train_Euclid_res_2000_TrainStars_id_001.npy \
    --base_path /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/phase_retrieval_paper/outputs/wf_output_benchmark_run/ \
    --metric_base_path /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/phase_retrieval_paper/outputs/wf_output_benchmark_run/metrics/ \
    --chkp_save_path /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/phase_retrieval_paper/outputs/wf_output_benchmark_run/chkp/ \
    --plots_folder plots/ \
    --model_folder chkp/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \
    --suffix_id_name v0 --suffix_id_name v1 --suffix_id_name v2 --suffix_id_name v3 \
    --star_numbers 10 --star_numbers 20 --star_numbers 30 --star_numbers 40 \
    ${opt[$SLURM_ARRAY_TASK_ID]} \


