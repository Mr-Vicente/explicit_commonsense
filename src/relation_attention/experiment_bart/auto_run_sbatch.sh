#!/bin/bash
#SBATCH --job-name=bart_swarm_piu_piu
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/error/%j-%x.err
#SBATCH --time=0 # No time limit
#SBATCH -n 5 # Number of tasks
#SBATCH --cpus-per-task 4 # number cpus per task
#SBATCH --mem=190000
#SBATCH --gres=gpu:3g.20gb:3

eval "$(conda shell.bash hook)"
# activate desired environment conda activate <env>
conda activate $1
# change dir to where we want to run scripts
cd ~/explicit_commonsense/src/relation_attention/experiment_bart &&

declare -a datasets=("eli5" "stackexchange_qa")

# default
args=(
--datasets "${datasets[@]}"
--experiment_type "default"
)
echo "running default"
sbatch -n1 -c4 --job-name="bart_default" --mem=30000 --time=0 --gres=gpu:3g.20gb:1 python $2 "${args[@]}" &

#declare -a run_types=("default" "relations" "mask")
# swow
declare -a kg_types=("conceptnet")
declare -a mask_types=("none" "random")
declare -a rel_types=("relations" "mask")

for kg_type in "${kg_types[@]}"
  do
    echo "running $kg_type"
    python3 pre_fast_tokenization.py --datasets "${datasets[@]}" --experiment_type "relations" --knowledge "$kg_type"
    for mask_type in "${mask_types[@]}"
      do
          echo "running $mask_type"
          args=(
            --datasets "${datasets[@]}"
            --experiment_type "relations"
            --knowledge "$kg_type"
            --head_mask_type "$mask_type"
          )
          sbatch -n1 -c4 --job-name="bart_${kg_type}_${mask_type}" --mem=80000 --time=0 --gres=gpu:3g.20gb:1 python $2 "${args[@]}" &
          sleep 3
      done
    sleep 3
    #python3 pre_fast_tokenization.py --datasets "${datasets[@]}" --experiment_type "mask" --knowledge "$kg_type" &
    #args=(
    #        --datasets "${datasets[@]}"
    #        --experiment_type "mask"
    #        --knowledge "$kg_type"
    #        --head_mask_type "random"
    #)
    #srun -n1 -c4 --mem=80000 --time=0 --gres=gpu:3g.20gb:1 python3 $2 "${args[@]}" &
    #sleep 3
  done
wait