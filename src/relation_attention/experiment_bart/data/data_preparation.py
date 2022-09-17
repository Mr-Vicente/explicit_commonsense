
#############################
#   Imports
#############################

# Python modules
from typing import List, Dict
from collections import deque
import random
import csv

# Remote modules
import torch

# Local modules
from utils import Data_Type, read_json_file_2_dict

#############################
#   Constants
#############################

SPLIT = {'train': 0, 'validation': 1, 'test': 2}

def split_data(total_data, val_percentage=0.05, test_percentage=0.05):
    size = len(total_data)
    eval_size = int(size * val_percentage + size * test_percentage)
    train_limit = size - eval_size
    val_limit = train_limit + (eval_size - int(size * test_percentage))
    train = total_data[:train_limit]
    validation = total_data[train_limit:val_limit]
    test = total_data[val_limit:]
    #random.shuffle(train)
    #random.shuffle(validation)
    #random.shuffle(test)
    return [train, validation, test]

def preprocess_data(data):
    for idx, d_unit in enumerate(data):
        d_unit['idx'] = idx
    #new_data = [(idx, d_unit) for idx, d_unit in enumerate(data)]
    random.shuffle(data)
    all_data = split_data(data)
    return all_data

def load_and_preprocess_data(data_types: List[Data_Type], limit=None):
    if not limit:
        all_data = load_data(data_types)
    else:
        all_data = load_data(data_types)[:limit]
    all_data = preprocess_data(all_data)
    # all_data -> [train_data, val_data, test_data]
    return all_data

# old : '/home/fm.vicente/data/qa_data'
def load_data(data_types: List[Data_Type], data_dir = '../../data'):
    all_data = deque()
    for data_type in data_types:
        if data_type.value == Data_Type.ELI5.value:
            print('eli5', flush=True)
            train_split = read_json_file_2_dict(f'train_eli5.json', store_dir=data_dir)
            validation_split = read_json_file_2_dict(f'validation_eli5.json', store_dir=data_dir)
            test_split = read_json_file_2_dict(f'test_eli5.json', store_dir=data_dir)
            data = train_split
            data.extend(validation_split)
            data.extend(test_split)
        elif data_type.value == Data_Type.ASK_SCIENCE.value:
            print('science', flush=True)
            train_split = read_json_file_2_dict(f'train_eli5.json', store_dir=data_dir)
            validation_split = read_json_file_2_dict(f'validation_eli5.json', store_dir=data_dir)
            test_split = read_json_file_2_dict(f'test_eli5.json', store_dir=data_dir)
            data = train_split
            data.extend(validation_split)
            data.extend(test_split)
        elif data_type.value == Data_Type.STACK_EXCHANGE.value:
            print('stackexchange_final', flush=True)
            data = read_json_file_2_dict('stackexchange_final.json', store_dir=data_dir)
        elif data_type.value == Data_Type.COMMONGEN_QA.value:
            data = read_json_file_2_dict('commongen_qa_final.json', store_dir=data_dir)[:5000]
            #print(data[:5])
        elif data_type.value == Data_Type.COMMONSENSE_QA.value:
            data = read_json_file_2_dict('commonsense_qa_final.json', store_dir=data_dir)
        elif data_type.value == Data_Type.COMMONGEN.value:
            data = read_json_file_2_dict('commongen.json', store_dir=data_dir)
        elif data_type.value == Data_Type.NATURAL_QUESTIONS.value:
            data = None
        else:
            raise NotImplementedError()
        all_data.extend(data)
    return list(all_data)

def load_csv_data_lazy(data_path):
    with open(data_path) as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            yield row

def load_csv_data(data_path):
    rows = deque()
    with open(data_path) as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            rows.append(row)
    return list(rows)

def load_tok_data(type_data, exp_type, store_dir, kg, datasets, use_context_str, use_extra_rels_str):
    path = f"{store_dir}/{datasets}/{exp_type}_{type_data}_{kg}{use_context_str}{use_extra_rels_str}.pkl"
    return torch.load(path)

def load_tok_data_simple(type_data, store_dir, datasets, use_context_str):
    path = f"{store_dir}/{datasets}/{type_data}{use_context_str}.pkl"
    return torch.load(path)
