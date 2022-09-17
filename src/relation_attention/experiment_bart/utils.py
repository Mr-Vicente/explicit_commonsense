#############################
#   Imports and Contants    #
#############################

# Python modules
from enum import Enum
import os
import json
import time
from dataclasses import dataclass

# Remote packages
import torch

#############################
#         utilities
#############################

class ScoringType(Enum):
    DEFAULT = 'default'
    MAX_PROB = 'max-prob'
    INTERPOL = 'interpol'
    CONSTRAINT = 'constraint'
    MULTIPLE_CHOICE = 'multiple_choice'

class LossType(Enum):
    DEFAULT = 'default'
    CP_RP_DEF = 'cp-rp-def'
    CP_DEF = 'cp-def'
    PRP_NRP_DEF = 'prp-nrp-def'

class Head_Mask(Enum):
    ALL = 'all'
    NONE = 'none'
    RANDOM = 'random'
    SPECIFIC = 'specific'

class KGType(Enum):
    SWOW = 'swow'
    CSKG = 'cskg'
    CONCEPTNET = 'conceptnet'

class Model_Type(Enum):
    RELATIONS = 'relations'
    MASK = 'mask'
    DEFAULT = 'default'

    def is_simple_mask_commonsense(self):
        return self == Model_Type.MASK

    def there_is_difference_between_relations(self):
        return self == Model_Type.RELATIONS

class Data_Type(Enum):
    ELI5 = 'eli5'
    COMMONSENSE_QA = 'commonsense_qa'
    COMMONGEN_QA = 'commongen_qa'
    STACK_EXCHANGE = 'stackexchange_qa'
    ASK_SCIENCE = 'ask_science_qa'
    NATURAL_QUESTIONS = 'natural_questions'
    LAMA = 'lama'
    CONCEPTNET = 'conceptnet'
    CUSTOM = 'custom'
    COMMONGEN = 'commongen'

    @staticmethod
    def data_types_to_str(data_types):
        datasets_str = '-'.join([x.value for x in data_types])
        return datasets_str

#############################
#         Models
#############################

MODELS_PRETRAINING_NAME = {
    "bart_large": "facebook/bart-large",
    "bart_large_fp32": "patrickvonplaten/bart-large-fp32",
    "bart_large_tweak": "",
    "bart_base": "facebook/bart-base"
}

CURRENT_PRETRAINING_NAME = MODELS_PRETRAINING_NAME.get('bart_large_fp32')

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
    elements = []
    with open(f'{store_dir}/{filename}', 'r', encoding='utf-8') as file:
        for line in file:
            elements.append(json.loads(line))
        return elements

def read_txt_2_list(filename, store_dir='.'):
    with open(f'{store_dir}/{filename}', 'r', encoding='utf-8') as file:
        return file.read().split('\n')

#############################
#           Data Structures helper functions
#############################

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    jump = len(lst)//n
    for i in range(0, len(lst), jump):
        yield lst[i:i + jump]

def get_jump_chunks(lst, jump):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), jump):
        yield lst[i:i + jump]

def join_str_first(sep_str, lis):
    return '{1}{0}'.format(sep_str.join(lis), sep_str).strip()

#############################
#           Huggingface
#############################

def inputs_introspection_print(tokenizer, inputs):
    input_ids = inputs.get('input_ids', None)
    input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    labels_ids = inputs.get('labels', None)
    labels_text = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
    print('orginal input:', input_text[:2])
    print("::::::::::::::::::::::::::")
    print('orginal labels:', labels_text[:2])
    print("==========|||||==========")

def tok_data_2_text(tokenizer, all_inputs):
    def clean_input_text(text):
        real_text = text.split(tokenizer.eos_token)[0]
        real_text = real_text.replace(tokenizer.bos_token, '').strip()
        return real_text
    all_input_text, all_labels_text = [], []
    for inputs in all_inputs:
        input_ids = inputs.get('input_ids', None)
        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        labels_ids = inputs.get('labels', None)
        labels_text = tokenizer.decode(labels_ids, skip_special_tokens=True)
        #print('input_text:', input_text)
        #print('labels_text:', labels_text)
        input_text = clean_input_text(input_text)
        all_input_text.append(input_text)
        all_labels_text.append(labels_text)
    return all_input_text, all_labels_text

#############################
#           Torch
#############################

def get_device(verbose:bool=True):
    # If there's a GPU available...
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        first_gpu = torch.cuda.get_device_name(0)
        if verbose:
            print(f'There are {n_gpus} GPU(s) available.')
            print(f'GPU gonna be used: {first_gpu}')
    else:
        if verbose:
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
