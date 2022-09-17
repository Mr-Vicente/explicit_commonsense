#!/bin/bash
#SBATCH --job-name=bart_csqa_rel
# The line below writes to a logs dir inside the one where sbatch was called # %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/error/%j-%x.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 4 # number cpus per task
#SBATCH --mem=30000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
# The line below is asking for for three 20 GBs GPUs
# gpu:7g.40gb:1 vs gpu:3g.20gb:1
# gpu:nvidia_a100-pcie-40gb | gpu:nvidia_a100-sxm4-40gb
# gpu:nvidia_a100-sxm4-40gb:4(S:0-1),shard:nvidia_a100-sxm4-40gb:28(S:0-1)
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1#,shard:7
eval "$(conda shell.bash hook)"
# activate desired environment conda activate <env>
conda activate $1
# change dir to where we want to run scripts
cd ~/explicit_commonsense/src/relation_attention/experiment_bart &&
# run program
#bash general_scripts/cluster/$3

declare -a datasets=("commonsense_qa")
declare -a kg_types=("conceptnet" "swow") &&
declare -a mask_types=("none" "random") &&

#PYTHONPATH=.:$PYTHONPATH python3 $2 --datasets "${datasets[@]}" --experiment_type "default" --learn_pos_embed_encoder "yes"\
#                                    --scoring_type "default" --loss_type "default" --use_context "yes" \
#                                    --pre_model "facebook/bart-large" --epochs 10 --batch_size 128 --max_length 64
PYTHONPATH=.:$PYTHONPATH python3 $2 --datasets "${datasets[@]}" --experiment_type "relations" --knowledge "conceptnet" --head_mask_type "none" \
                                    --scoring_type "default" --loss_type "default" --learn_pos_embed_encoder "yes" --use_context "yes" --use_extra_relations "no" \
                                    --pre_model "facebook/bart-large" --epochs 15 --batch_size 128 --max_length 64
#PYTHONPATH=.:$PYTHONPATH python3 $2 --datasets "${datasets[@]}" --experiment_type "mask" --knowledge "conceptnet" --head_mask_type "random"