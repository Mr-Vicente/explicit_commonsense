#############################
#   Imports
#############################

# Python modules
from typing import List
import csv
from collections import defaultdict
import argparse
import random

# Remote modules

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

def get_args():
    print('-----Argument parsing------')
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=None, help="Data containing (id, concepts, labels)")
    parser.add_argument("--models_names", nargs='+', type=str, default=None, help="datasets to train")

    args = parser.parse_args()
    return args


def load_data(data_path):
    with open(f'data/test_data/{data_path}.csv', 'r') as f:
        data_reader = csv.reader(f)
        data = []
        for d in data_reader:
            data.append(d)
        return data

def load_model_gens_data(models_names:List[str]):
    models_gens = defaultdict(list)
    for model_name in models_names:
        with open(f'data/model_gens/{model_name}.csv', 'r') as f:
            model_gens_reader = csv.reader(f)
            for d in model_gens_reader:
                input_id, gen = d
                models_gens[f'{input_id}'].append(gen)
    return models_gens

def load_model_absqa_gens_data(models_names:List[str]):
    models_gens = defaultdict(list)
    for model_name in models_names:
        with open(f'data/model_gens/{model_name}.csv', 'r') as f:
            model_gens_reader = csv.reader(f)
            for d in model_gens_reader:
                input_id, gen = d
                models_gens[f'{input_id}'].append(gen)
    return models_gens

def create_hit(hit_name, data, models_gens, models_names):
    concepts_graber_ids = list(range(len(data)))
    random.shuffle(concepts_graber_ids)
    #step1
    commonsense_column_names = [f'model_{i}' for i in range(len(models_names))]
    #step2
    description_column_names = [f'model_{i}' for i in range(len(models_names))]
    #step3
    concepts_column_names = ['step3_concepts']
    #step4
    commonsense_column_names = [f'model_{i}' for i in range(len(models_names))]
    with open(f'{hit_name}.csv', 'w') as f:
        hit_writer = csv.writer(f)
        hit_writer.writerow(['test_example_i', *models_names, 'concepts'])
        for (c_id, (i_id, model_gen)) in zip(concepts_graber_ids, models_gens.items()):
            i_id = int(i_id)
            input_example = data[c_id]
            concepts_from_example = input_example[1]
            hit_writer.writerow([i_id, *model_gen, concepts_from_example])

if __name__ == '__main__':
    args = get_args()
    models_names = args.models_names
    data_path = args.data_path
    data = load_data(data_path=data_path)
    models_gens = load_model_gens_data(models_names=models_names)
    #data = [[0,'gato pato cao',''],[1,'piu raw cao',''],[2,'=====',''],[3,'o ps d','']]
    #models_gens = {'3': ["meow", "raw"], '1': ["meow", "raw"], '2': ["meow", "raw"],'0': ["meow", "raw"]}
    #models_names = ["patusco", "pixy"]
    create_hit('test', data, models_gens, models_names)