#!/bin/bash
#SBATCH --job-name=c_predictions
# The line below writes to a logs dir inside the one where sbatch was called # %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/error/%x-%j.out
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 4 # number cpus per task
#SBATCH --mem=30000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
# The line below is asking for for three 20 GBs GPUs
# gpu:7g.40gb:1 vs gpu:3g.20gb:1
# gpu:nvidia_a100-pcie-40gb | gpu:nvidia_a100-sxm4-40gb
# gpu:nvidia_a100-sxm4-40gb:4(S:0-1),shard:nvidia_a100-sxm4-40gb:28(S:0-1)
#SBATCH --gres=shard:5
eval "$(conda shell.bash hook)"
# activate desired environment conda activate <env>
conda activate $1
# change dir to where we want to run scripts
cd ~/explicit_commonsense/src/relation_attention/experiment_bart &&
# run program
#bash general_scripts/cluster/$3
declare -a DF_COM_MODEL_PATHS=("/home/fm.vicente/data/models_weights/trained_models_3/facebook-bart-large_default_commongen_none_none_L-default_DS-default_wLearnEmb_2/checkpoint-2331") &&
declare -a REL_COM_MODEL_NAMES=("/home/fm.vicente/data/models_weights/trained_models_2/facebook-bart-large_relations_commongen_conceptnet_none_L-default_DS-default_wLearnEmb_4/checkpoint-4653" "/home/fm.vicente/data/models_weights/trained_models_2/facebook-bart-large_relations_commongen_conceptnet_none_L-cp-rp-def_DS-default_wLearnEmb_7/checkpoint-4653" "/home/fm.vicente/data/models_weights/trained_models_3/facebook-bart-large_relations_commongen_conceptnet_none_L-default_DS-constraint_wExtraRels_wLearnEmb_5/checkpoint-3102") &&

declare -a DF_ABS_MODEL_PATHS=("/home/fm.vicente/data/models_weights/trained_models_3/facebook-bart-large_default_eli5-stackexchange_qa_conceptnet_none_L-default_DS-default_wLearnEmb_3/checkpoint-35567") &&
declare -a REL_ABS_MODEL_NAMES=("/home/fm.vicente/data/models_weights/trained_models_3/facebook-bart-large_relations_eli5-stackexchange_qa_conceptnet_none_L-default_DS-default_wLearnEmb_10/checkpoint-30804") &&

#declare -a DF_COM_MODEL_PATHS=("") &&
#declare -a REL_COM_MODEL_NAMES=("") &&

#declare -a DF_ABS_MODEL_PATHS=("") &&
#declare -a REL_ABS_MODEL_NAMES=("") &&


PYTHONPATH=.:$PYTHONPATH python3 create_predictions.py --default_commongen_models_paths "${DF_COM_MODEL_PATHS[@]}"  \
                            --relations_commongen_models_paths "${REL_COM_MODEL_NAMES[@]}"  \
                            --default_absqa_models_paths "${DF_ABS_MODEL_PATHS[@]}"  \
                            --relations_absqa_models_paths "${REL_ABS_MODEL_NAMES[@]}"