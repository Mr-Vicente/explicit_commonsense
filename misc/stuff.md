### Run on cluster (example)
sbatch ~/VCSG/general_scripts/cluster/run_on_cluster.sh curiosities_matching vcsg curiosities_matching.sh
sbatch ~/NS-BART/src/ns_bart/CSBart/relation_attention/run_on_cluster.sh ns_bart train_bart.py

### transfer data (example)
scp -r ~/Documents/Education/Thesis/code/old/articles_decomposition fm.vicente@ncluster:/home/fm.vicente/thesis_datasets/articles_decomposition

## CHANGE JUPYTER NOTEBOOK THEMES
# install
conda install -c conda-forge jupyterthemes
# list themes
jt -l
# choose theme
jt -t theme-name
# restart kernel if needed

## Prepare environment for jupypter notebooks
conda activate [env_name]
conda install -c anaconda ipykernel
/opt/anaconda3/envs/[env_name]/bin/python3 -m ipykernel install --user --name=[env_name]

### shh agent config
# go to bash console
eval `ssh-agent`
ssh-add my_cluster_rsa_key

### pip instal to conda env
/opt/anaconda3/envs/[env]/bin/python3 -m spacy download en_core_web_lg
/opt/anaconda3/envs/Thoughts/bin/python3 -m pip install sense2vec

### Export Env
conda env export | grep -v "^prefix: " > environment.yml