
#############################
#   Imports
#############################

# Python modules
from typing import List, Dict
from collections import deque

# Remote modules
from torch.utils.data import Dataset
from tqdm import tqdm

# Local modules
from data.datasets_model_handling import DatasetParsingUtils

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class DatasetGeneral(Dataset):
    SPLIT = {'train': 0, 'validation': 1, 'test': 2}
    def __init__(
            self,
            datasets_parsing_utils:DatasetParsingUtils,
            tokenizer=None,
            device=None,
            max_length=128
    ):
        print('initiated dataset')
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.device = device
        print('Loading dataset...')
        self.data = datasets_parsing_utils.get_data()
        print('tokenizing dataset')
        self.tokenized_data = self.tokenize_data()

    def tokenize_data(self):
        tokenized_data = deque()
        for idx, _d in tqdm(enumerate(self.data)):
            tokenized_data.append(self.make_example(idx))
        return list(tokenized_data)

    def get_sample(self, idx):
        example = self.data[idx]
        dataset_idx = example["idx"]
        input_data = example["input_data"]
        label_data = example["labels_data"]
        input_data_sample = input_data.lower()
        label_sample = label_data.lower()
        return input_data_sample, label_sample, dataset_idx

    def transform_tokenization(self, source, target) -> dict:
        source_ids = source["input_ids"].squeeze()
        target_ids = target["input_ids"].squeeze()
        #target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        src_mask = source["attention_mask"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        data_example = {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
            "decoder_attention_mask": target_mask
        }
        return data_example

    def make_example(self, idx) -> dict:
        question_sample, answer_sample, _ = self.get_sample(idx)
        source = self.tokenizer(question_sample, padding='max_length',
                                truncation='longest_first',  max_length=self.max_length,
                                return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer(answer_sample, padding='max_length',
                                    truncation='longest_first', max_length=self.max_length,
                                    return_tensors="pt")

        data_example = self.transform_tokenization(source, target)
        return data_example

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def __len__(self):
        return len(self.tokenized_data)