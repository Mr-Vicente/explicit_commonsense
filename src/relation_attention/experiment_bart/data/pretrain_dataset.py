
#############################
#   Imports
#############################

# Python modules
from typing import List, Dict
from collections import deque

# Remote modules
import numpy as np

# Local modules
from .DatasetGeneral import DatasetGeneral
from torch.utils.data import Dataset
from .preprocessing import clean_mask_labels

from utils import Data_Type

from .relation_utils import clean_relations

#############################
#   Constants
#############################

#############################
#   Stuff
#############################


class PretrainDataset(Dataset):
    def __init__(
            self,
            data,
            data_type: Data_Type,
            tokenizer=None,
            max_length=128,
            mask_token='<mask>'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token = mask_token
        self.data = self.transform_data(data, data_type)
        self.tokenized_data = self.tokenize_data()

    def transform_data(self, data, data_type: Data_Type):
        new_data = deque()
        if data_type.value == Data_Type.CONCEPTNET.value:
            for data_point in data:
                if len(data_point)>2:
                    continue
                masked_data, label_data = data_point
                if self.mask_token not in masked_data:
                    continue
                new_data_point = {
                    "input": masked_data,
                    "label": label_data,
                }
                new_data.append(new_data_point)
        else:
            raise NotImplementedError()
        return list(new_data)

    def tokenize_data(self):
        tokenized_data = deque()
        for idx, _ in enumerate(self.data):
            tokenized_data.append(self.make_example(idx))
        return list(tokenized_data)

    def _get_sample(self, idx):
        data_point: Dict = self.data[idx]
        input_sentence = data_point["input"]
        label = data_point["label"]
        input_sample = input_sentence.lower()
        label_sample = label.lower()
        return input_sample, label_sample

    def transform_tokenization(self, source, target) -> dict:
        source_ids = source["input_ids"].squeeze()
        target_ids = target["input_ids"].squeeze()

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
        input_sample, labels_sample = self._get_sample(idx)
        source = self.tokenizer(input_sample, padding='max_length',
                                truncation='longest_first',
                                return_tensors="pt", max_length=self.max_length)
        target = self.tokenizer(labels_sample, padding='max_length',
                                truncation='longest_first',
                                return_tensors="pt", max_length=self.max_length)

        data_example = self.transform_tokenization(source, target)
        return data_example

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def __len__(self):
        return len(self.tokenized_data)