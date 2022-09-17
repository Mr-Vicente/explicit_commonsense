#!/bin/bash
#SBATCH --job-name=ev_csqa
# The line below writes to a logs dir inside the one where sbatch was called # %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/error/%j-%x.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 4 # number cpus per task
#SBATCH --mem=80000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
# The line below is asking for for three 20 GBs GPUs
# gpu:7g.40gb:1 vs gpu:3g.20gb:1
#SBATCH --gres=gpu:3g.20gb:1 # GPU Options gpu:3g.20gb:<1-16>
eval "$(conda shell.bash hook)"
# activate desired environment conda activate <env>
conda activate $1
# change dir to where we want to run scripts
cd ~/explicit_commonsense/src/relation_attention/experiment_bart &&
# run program
#bash general_scripts/cluster/$3

#declare -a kg_types=("conceptnet" "swow") &&
#declare -a mask_types=("none" "random") &&

python3 $2 --dataset "commonsense_qa" --experiment_type "default" --model_path "./trained_models/default_none_none_2_facebook-bart-large_3e-05_16/checkpoint-198700" &&
python3 $2 --dataset "commonsense_qa" --experiment_type "relations" --knowledge "conceptnet" --model_path "./trained_models/relations_conceptnet_none_22_facebook-bart-large_3e-05_72/checkpoint-43820" &&
python3 $2 --dataset "commonsense_qa" --experiment_type "mask" --knowledge "conceptnet" --model_path "./trained_models/mask_conceptnet_random_2_facebook-bart-large_3e-05_72/checkpoint-43820" &&
cd eval &&
python3 evals_to_plots.py --dataset "commonsense_qa"
#python3 $2 --experiment_type "relations" --knowledge "swow" --model_path ""
#python3 $2 --experiment_type "mask" --knowledge "swow" --model_path ""