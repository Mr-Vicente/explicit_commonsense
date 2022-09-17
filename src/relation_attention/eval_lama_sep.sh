#!/bin/bash
#SBATCH --job-name=ev_bm
# The line below writes to a logs dir inside the one where sbatch was called # %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/error/%j-%x.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 4 # number cpus per task
#SBATCH --mem=40000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
# The line below is asking for for three 20 GBs GPUs
# gpu:7g.40gb:1 vs gpu:3g.20gb:1
#SBATCH --gres=shard:4
eval "$(conda shell.bash hook)"
# activate desired environment conda activate <env>
conda activate $1
# change dir to where we want to run scripts
cd ~/explicit_commonsense/src/relation_attention/experiment_bart &&
# run program
#bash general_scripts/cluster/$3

#python3 $2 --experiment_type "default" --model_path "facebook/bart-large" &&
#python3 $2 --experiment_type "default" --model_path "./trained_models/facebook-bart-large_default_commongen_none_none_L-default_DS-default_wLearnEmb_2/checkpoint-2590" &&
python3 $2 --experiment_type "relations" --knowledge "conceptnet" --model_path "/home/fm.vicente/data/models_weights/trained_models_2/facebook-bart-large_relations_commongen_conceptnet_none_L-cp-rp-def_DS-default_wLearnEmb_7/checkpoint-4653" #&&
#python3 $2 --experiment_type "default" --model_path "./trained_models/default_commongen_none_none_2_3e-05_256/checkpoint-2590" &&
#python3 $2 --experiment_type "relations" --knowledge "conceptnet" --model_path "./trained_models/relations_commongen_conceptnet_none_4_3e-05_128/checkpoint-5170" &&
#python3 $2 --experiment_type "mask" --knowledge "conceptnet" --model_path "./trained_models/mask_commongen_conceptnet_random_4_3e-05_128/checkpoint-5170" &&
#PYTHONPATH=.:$PYTHONPATH python3 eval/evals_to_plots.py --dataset "lama"
