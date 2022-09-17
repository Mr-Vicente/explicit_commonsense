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

    parser.add_argument("--concepts_data_path", type=str, default="concepts.txt", help="")
    parser.add_argument("--absqa_models_paths", nargs='+', type=str, default=None, help="")
    parser.add_argument("--commongen_models_paths", nargs='+', type=str, default=None, help="")

    args = parser.parse_args()
    return args


def load_data(data_path):
    with open(f'data/test_data/{data_path}.csv', 'r') as f:
        data_reader = csv.reader(f)
        data = []
        for d in data_reader:
            data.append(d)
        return data

def load_commongen_model_gens_data(models_names:List[str]):
    models_gens = defaultdict(list)
    for model_path in models_names:
        if 'checkpoint' in model_path:
            model_path = '/'.join(model_path.split('/')[:-1])
        with open(f'{model_path}/predictions.csv', 'r') as f:
            model_gens_reader = csv.reader(f)
            for i, d in enumerate(model_gens_reader):
                if i == 200:
                    break
                inp, pred, label = d
                #models_gens[f'{i}'].append(pred)
                models_gens[f'{i}'].append({'input': inp, 'pred': pred})
    return models_gens

def create_hit(hit_name, commongen_inputs, commongen_preds, absqa_qas, models_names, absqa_models_names):
    #step1
    commonsense_column_names = [f'commonsense_model_{i}' for i in range(len(models_names))]
    #step2
    #description_column_names = [f'descriptive_model_{i}' for i in range(len(models_names))]
    #step3
    concepts_column_names = ['step3_concepts']
    #step4
    absqa_column_q_names = [f'absqa_model_q_{i}' for i in range(len(absqa_models_names))]
    absqa_column_a_names = [f'absqa_model_a_{i}' for i in range(len(absqa_models_names))]
    with open(f'{hit_name}.csv', 'w') as f:
        hit_writer = csv.writer(f)
        hit_writer.writerow([#'test_example_i',
                             *commonsense_column_names,
                             #*description_column_names,
                             *concepts_column_names,
                             *absqa_column_q_names,
                            *absqa_column_a_names])
        for row_concepts, commongen_data, absqa_data in zip(commongen_inputs, commongen_preds.items(), absqa_qas.items()):
            i_id = commongen_data[0]
            commongen_model_results = commongen_data[-1]
            commongen_model_results_phrases = [line['pred'] for line in commongen_model_results]
            absqa_model_results = absqa_data[-1]
            #print('absqa_model_results:', absqa_model_results)
            absqa_model_results_qs = [line['input'] for line in absqa_model_results]
            #print('absqa_model_results_qs:', absqa_model_results_qs)
            absqa_model_results_as = [line['pred'] for line in absqa_model_results]
            #print('absqa_model_results_as:', absqa_model_results_as)
            hit_writer.writerow([*commongen_model_results_phrases, #*commongen_model_results_phrases,
                                 row_concepts, *absqa_model_results_qs, *absqa_model_results_as])

if __name__ == '__main__':
    args = get_args()
    concepts_path = args.concepts_data_path
    with open(concepts_path, 'r')as f:
        concepts_list = f.readlines()
        concepts_list = [concepts.replace('\n','') for concepts in concepts_list]
        random.shuffle(concepts_list)
    commongen_models_paths = args.commongen_models_paths
    absqa_models_paths = args.absqa_models_paths
    print(commongen_models_paths)
    commongen_preds = load_commongen_model_gens_data(models_names=commongen_models_paths)
    absqa_qas = load_commongen_model_gens_data(models_names=absqa_models_paths)
    print('concepts:',concepts_list)
    create_hit('test', concepts_list, commongen_preds, absqa_qas, commongen_models_paths, absqa_models_paths)