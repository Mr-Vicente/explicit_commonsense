
#############################
#   Imports
#############################

# Python modules
from typing import Dict
from collections import deque

# Remote modules
import numpy as np

# Local modules
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


class EvalDataset(Dataset):
    SPLIT = {'train': 0, 'validation': 1, 'test': 2}
    def __init__(
            self,
            data,
            data_type: Data_Type,
            tokenizer=None,
            device=None,
            max_length=32,
    ):
        print('initiated dataset')
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.device = device
        self.data = self.transform_data(data, data_type)
        print('tokenizing dataset')
        self.tokenized_data = self.tokenize_data()

    def transform_data(self, data, data_type: Data_Type):
        new_data = deque()
        if data_type.value == Data_Type.LAMA.value:
            for data_point in data:
                masked_sentence = data_point.get('masked_sentences', [""])[0]
                if len(masked_sentence.split(' ')) >= 80:
                    continue
                masked_sentence = clean_mask_labels(masked_sentence)
                label = data_point.get('obj_label', data_point.get('sub_label', ""))
                new_data_point = {
                    "input": masked_sentence,
                    "label": label,
                }
                new_data.append(new_data_point)
        elif data_type.value == Data_Type.COMMONSENSE_QA.value:
            LABELS = ['A', 'B', 'C', 'D', 'E']
            for data_point in data:
                _qid = data_point['id']
                question = data_point['question']['stem']
                # garantee the order of labels A,B,C,D,E and obtain corresponding answers
                answers = np.array(
                    [choice['text'] for choice in sorted(data_point['question']['choices'], key=lambda c: c['label'])])
                # the test set has no answer key so use 'A' as a dummy label
                # label = self.LABELS.index(line.get('answerKey', 'A'))
                label = answers[LABELS.index(data_point.get('answerKey', 'A'))]
                new_data_point = {
                    "input": question,
                    "label": label,
                }
                new_data.append(new_data_point)
        elif data_type.value == Data_Type.CUSTOM.value:
            for data_point in data:
                masked_data, label_data = data_point
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


class EvalRelationsDataset(EvalDataset):
    def __init__(
            self,
            data,
            relation_mapper_builder,
            data_type: Data_Type,
            tokenizer=None,
            device=None,
            max_length=32,
    ):
        self.relation_mapper_builder = relation_mapper_builder
        super(EvalRelationsDataset, self).__init__(
            data=data,
            data_type=data_type,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length
        )

    def pre_process_context(self, context):
        context = context.lower()
        # process context in search for relations
        commonsense_relations = self.relation_mapper_builder.get_relations_mapping_complex(context=[context])
        # clean relation
        commonsense_relation = clean_relations(commonsense_relations)[0]
        # convert this relations to matrices
        print(commonsense_relation)
        context_tokenized = self.tokenizer(context, padding='max_length',
                                truncation='longest_first',  max_length=self.max_length,
                                return_tensors="pt", return_offsets_mapping=True,
                                input_commonsense_relations=commonsense_relation,
                                )
        return context_tokenized

    def make_example(self, idx) -> dict:
        input_sample, labels_sample = self._get_sample(idx)
        source = self.pre_process_context(input_sample)
        target = self.tokenizer(labels_sample, padding='max_length',
            truncation='longest_first', max_length=self.max_length,
            input_commonsense_relations=None, return_offsets_mapping=True,
            return_tensors="pt"
        )
        source_input_relations = source["input_commonsense_relations"].squeeze()
        data_example = self.transform_tokenization(source, target)
        data_example['input_commonsense_relations'] = source_input_relations
        return data_example