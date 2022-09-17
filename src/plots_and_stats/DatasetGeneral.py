
#############################
#   Imports
#############################

# Python modules
from typing import List, Dict
from collections import deque
import random

# Remote modules
from torch.utils.data import Dataset
from tqdm import tqdm

# Local modules

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
            data,
            tokenizer=None,
            extra_answer_threshold=0,
            device=None,
            max_length=128
    ):
        print('initiated dataset')
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.device = device
        print('dataset loaded')
        self.data, self.qa_id_list = self.get_helper_index(data, extra_answer_threshold)
        print('tokenizing dataset')
        self.tokenized_data = self.tokenize_data()

    def get_helper_index(self, data, extra_answer_threshold):
        #print('self.data: ', self.data)
        new_data = deque()
        qa_id_list = deque()
        counter = 0
        for idx, qa in data:
            usefull = False
            for j, (a, sc) in enumerate(zip(qa["answers"]["text"], qa["answers"]["score"])):
                if j == 0 and int(sc) >= extra_answer_threshold:
                    usefull = True
                    qa_id_list.append((counter, j))
            if usefull:
                new_data.append((idx, qa))
                counter+=1
        print(len(new_data), len(qa_id_list))
        return list(new_data), list(qa_id_list)

    def tokenize_data(self):
        tokenized_data = deque()
        for idx, (_i, _qa) in enumerate(self.data):
            tokenized_data.append(self.make_example(idx))
        return list(tokenized_data)

    def clean_data(self, data):
        return data

    def _get_sample(self, idx):
        i, j = self.qa_id_list[idx]
        _dataset_idx, example = self.data[i]
        question = example["title"]
        answer = example["answers"]["text"][j]
        question_sample = question.lower()
        answer_sample = answer.lower()
        return question_sample, answer_sample, _dataset_idx

    def transform_tokenization(self, source, target) -> dict:
        source_ids = source["input_ids"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_ids[target_ids == 0] = -100

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
        question_sample, answer_sample, _ = self._get_sample(idx)
        source = self.tokenizer(question_sample, padding='max_length',
                                truncation='longest_first',  max_length=self.max_length,
                                return_tensors="pt")
        target = self.tokenizer(answer_sample, padding='max_length',
                                truncation='longest_first', max_length=self.max_length,
                                return_tensors="pt")

        data_example = self.transform_tokenization(source, target)
        return data_example

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def __len__(self):
        return len(self.tokenized_data)