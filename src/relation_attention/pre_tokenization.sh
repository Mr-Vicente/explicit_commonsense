#!/bin/bash
#SBATCH --job-name=pre_tok
# The line below writes to a logs dir inside the one where sbatch was called # %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/error/%j-%x.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 4 # number cpus per task
#SBATCH --mem=60000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
# The line below is asking for for three 20 GBs GPUs
# gpu:7g.40gb:1 vs gpu:3g.20gb:1
eval "$(conda shell.bash hook)"
# activate desired environment conda activate <env>
conda activate $1
# change dir to where we want to run scripts
cd ~/explicit_commonsense/src/relation_attention/experiment_bart &&
# run program
#bash general_scripts/cluster/$3

#PYTHONPATH=.:$PYTHONPATH python3 pre_fast_tokenization.py --datasets "${datasets[@]}" --experiment_type "mask" --knowledge "conceptnet" --max_length 32 \
#                                                          --use_extra_relations "yes" --use_context "yes" --pre_training_model "facebook/bart-large"
#no extra relations
#PYTHONPATH=.:$PYTHONPATH python3 pre_fast_tokenization.py --datasets "${datasets[@]}" --experiment_type "mask" --knowledge "conceptnet" --max_length 32 \
#                                                          --use_extra_relations "no" --use_context "no" --pre_training_model "facebook/bart-large"
#PYTHONPATH=.:$PYTHONPATH python3 pre_fast_tokenization.py --datasets "${datasets[@]}" --experiment_type "mask" --knowledge "conceptnet" --max_length 32 \
#                                                          --use_extra_relations "no" --use_context "no" --pre_training_model "facebook/bart-large"


# Commonsenseqa
declare -a datasets=("commonsense_qa") &&
PYTHONPATH=.:$PYTHONPATH python3 pre_fast_tokenization.py --datasets "${datasets[@]}" --experiment_type "relations" --knowledge "conceptnet" --max_length 64 \
                                                          --use_extra_relations "no" --use_context "yes" --pre_training_model "facebook/bart-large" --should_tokenize_default "yes"

# eli5
#declare -a datasets=("eli5" "stackexchange_qa") &&
#PYTHONPATH=.:$PYTHONPATH python3 pre_fast_tokenization.py --datasets "${datasets[@]}" --experiment_type "relations" --knowledge "conceptnet" --max_length 128 \
#                                                          --use_extra_relations "no" --use_context "no" --pre_training_model "facebook/bart-large"

# commongen
#declare -a datasets=("commongen") &&
#PYTHONPATH=.:$PYTHONPATH python3 pre_fast_tokenization.py --datasets "${datasets[@]}" --experiment_type "relations" --knowledge "conceptnet" --max_length 32 \
#                                                          --use_extra_relations "yes" --use_context "no" --pre_training_model "facebook/bart-large"

