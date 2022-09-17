#!/bin/bash
#SBATCH --job-name=bart_com_default
# The line below writes to a logs dir inside the one where sbatch was called # %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/error/%j-%x.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 4 # number cpus per task
#SBATCH --mem=30000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
# The line below is asking for for three 20 GBs GPUs
# gpu:7g.40gb:1 vs gpu:3g.20gb:1
#SBATCH --gres=gpu:7g.40gb:1 # GPU Options gpu:3g.20gb:<1-16>
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1#,shard:7
eval "$(conda shell.bash hook)"
# activate desired environment conda activate <env>
conda activate $1
# change dir to where we want to run scripts
cd ~/explicit_commonsense/src/relation_attention/experiment_bart &&
# run program
#bash general_scripts/cluster/$3

declare -a datasets=("commongen")
#API_KEY="$(cat ./run_config/comet_api_key.txt)"            #cometml
#PROJECT_NAME="bart-commonsense"
python3 $2 --datasets "${datasets[@]}" --experiment_type "default" \
            --scoring_type "default" --loss_type "default" --learn_pos_embed_encoder "yes" \
            --use_context "no" --use_extra_relations "no" --epochs 10 \
            --pre_model "facebook/bart-large" --batch_size 256