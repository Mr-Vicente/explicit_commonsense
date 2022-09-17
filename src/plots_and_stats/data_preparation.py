
#############################
#   Imports
#############################

# Python modules
from typing import List, Dict
from collections import deque
import random

# Remote modules
from torch.utils.data import Dataset

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
    return [train, validation, test]

def preprocess_data(data):
    new_data = [(idx, d_unit) for idx, d_unit in enumerate(data)]
    random.shuffle(new_data)
    all_data = split_data(new_data)
    return all_data

def load_and_preprocess_data(data_types: List[Data_Type]):
    all_data = load_data(data_types)
    print('len:', len(all_data))
    print('all_data[0]: ', all_data[0])
    all_data = preprocess_data(all_data)
    return all_data

#def load_data(data_types: List[Data_Type], data_dir = '/home/fm.vicente/data/qa_data'):
def load_data(data_types: List[Data_Type], data_dir='/Users/mrvicente/Documents/Education/Thesis/code/f_papers/explicit_commonsense/src/relation_attention/'):
    all_data = deque()
    for data_type in data_types:
        if data_type == Data_Type.ELI5:
            print('eli5', flush=True)
            train_split = read_json_file_2_dict(f'train_eli5.json', store_dir=data_dir)
            validation_split = read_json_file_2_dict(f'validation_eli5.json', store_dir=data_dir)
            test_split = read_json_file_2_dict(f'test_eli5.json', store_dir=data_dir)
            data = train_split
            data.extend(validation_split)
            data.extend(test_split)
        elif data_type == Data_Type.ASK_SCIENCE:
            print('science', flush=True)
            train_split = read_json_file_2_dict(f'train_eli5.json', store_dir=data_dir)
            validation_split = read_json_file_2_dict(f'validation_eli5.json', store_dir=data_dir)
            test_split = read_json_file_2_dict(f'test_eli5.json', store_dir=data_dir)
            data = train_split
            data.extend(validation_split)
            data.extend(test_split)
        elif data_type == Data_Type.STACK_EXCHANGE:
            print('stackexchange_final', flush=True)
            data = read_json_file_2_dict('stackexchange_final.json', store_dir=data_dir)
        elif data_type.value == Data_Type.COMMONGEN_QA.value:
            data = read_json_file_2_dict('commongen_qa_final.json', store_dir=data_dir)[:5000]
        elif data_type == Data_Type.COMMONSENSE_QA:
            data = read_json_file_2_dict('commonsense_qa_final.json', store_dir=data_dir)
        elif data_type == Data_Type.NATURAL_QUESTIONS:
            data = None
        else:
            raise NotImplementedError()
        all_data.extend(data)
    return list(all_data)
