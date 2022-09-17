#############################
#   Imports and Contants    #
#############################

# Python modules
from enum import Enum
import os
import json
import time

# Remote packages
import torch

#############################
#         utilities
#############################

class Task_Type(Enum):
    COMMONSENSE = 'commonsense'

class Data_Type(Enum):
    ELI5 = 'eli5'
    COMMONSENSE_QA = 'commonsense_qa'
    COMMONGEN_QA = 'commongen_qa'
    STACK_EXCHANGE = 'stackexchange_qa'
    ASK_SCIENCE = 'ask_science_qa'

#############################
#   Files Managment         #
#############################

def create_directory(output_dir):
    # Create output directory if needed
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except FileExistsError as _:
            return
    else:
        print(f"Output directory {output_dir} already exists")

def read_simple_text_file_2_vec(filename, store_dir='.'):
    with open(f'{store_dir}/{filename}', 'r') as f:
        return f.read().split('\n')

def write_dict_2_json_file(json_object, filename, store_dir='.'):
    create_directory(store_dir)
    with open(f'{store_dir}/{filename}', 'w', encoding='utf-8') as file:
        json.dump(json_object, file, ensure_ascii=False, indent=4)


def read_json_file_2_dict(filename, store_dir='.'):
    with open(f'{store_dir}/{filename}', 'r', encoding='utf-8') as file:
        return json.load(file)

def read_jsonl_file_2_dict(filename, store_dir='.'):
    recipes = []
    with open(f'{store_dir}/{filename}', 'r', encoding='utf-8') as file:
        for line in file:
            recipes.append(json.loads(line))
        return recipes

#############################
#           Data Structures helper functions
#############################

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    jump = len(lst)//n
    for i in range(0, len(lst), jump):
        yield lst[i:i + jump]

#############################
#           Torch
#############################

def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        first_gpu = torch.cuda.get_device_name(0)

        print(f'There are {n_gpus} GPU(s) available.')
        print(f'GPU gonna be used: {first_gpu}')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

#############################
#         Timing
#############################

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        original_return_val = func(*args, **kwargs)
        end = time.time()
        print("time elapsed in ", func.__name__, ": ", end - start, sep='')
        return original_return_val

    return wrapper

#############################
#         PRINTING UTILS
#############################

class LOGGER_COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    INFOCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_info(logger, message):
    logger.info(f'{LOGGER_COLORS.INFOCYAN}[INFO]{LOGGER_COLORS.ENDC}: {message}')

def print_success(logger, message):
    logger.info(f'{LOGGER_COLORS.OKGREEN}[SUCCESS]{LOGGER_COLORS.ENDC}: {message}')

def print_warning(logger, message):
    logger.info(f'{LOGGER_COLORS.WARNING}[WARNING]{LOGGER_COLORS.ENDC}: {message}')

def print_fail(logger, message):
    logger.info(f'{LOGGER_COLORS.FAIL}[FAIL]{LOGGER_COLORS.ENDC}: {message}')
