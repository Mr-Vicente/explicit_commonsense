
#############################
#   Imports
#############################

# Python modules

# Remote modules

# Local modules
from relation_utils import clean_relations
from utils import read_json_file_2_dict
from DatasetGeneral import DatasetGeneral

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class RelationsDataset(DatasetGeneral):
    def __init__(
            self,
            data,
            tokenizer=None,
            extra_answer_threshold=1,
            device=None,
            max_length=128,
            word_relations_filename='relation_data.json',
            word_relations_data_dir='/Users/mrvicente/Documents/Education/Thesis/code/f_papers/explicit_commonsense/src/relation_attention',
            mask_heads=None
    ):
        word_relations = read_json_file_2_dict(word_relations_filename, store_dir=word_relations_data_dir)
        self.word_relations = clean_relations(word_relations)
        self.mask_heads = mask_heads
        super(RelationsDataset, self).__init__(
            data,
            tokenizer=tokenizer,
            extra_answer_threshold=extra_answer_threshold,
            device=device,
            max_length=max_length,
        )

    def make_example(self, idx) -> dict:
        question_sample, answer_sample, dataset_idx = self._get_sample(idx)
        #print('dataset_idx: ', dataset_idx)
        #print('len(self.word_relations): ', len(self.word_relations))
        #print('len(self.data): ', len(self.data))
        commonsense_relation = self.word_relations[dataset_idx]
        source = self.tokenizer(question_sample, padding='max_length',
                                truncation='longest_first',  max_length=self.max_length,
                                return_tensors="pt", return_offsets_mapping=True,
                                input_commonsense_relations=commonsense_relation,
                                )
        target = self.tokenizer(answer_sample, padding='max_length',
                                truncation='longest_first', max_length=self.max_length,
                                input_commonsense_relations=None, return_offsets_mapping=True,
                                return_tensors="pt")

        source_input_relations = source["input_commonsense_relations"].squeeze()
        data_example = self.transform_tokenization(source, target)
        data_example['input_commonsense_relations'] = source_input_relations
        if self.mask_heads is not None:
            # encoder mask only
            data_example['head_mask'] = self.mask_heads
        return data_example


class MaskRelationsDataset(RelationsDataset):
    def __init__(self,
            data,
            tokenizer=None,
            extra_answer_threshold=1,
            device=None,
            max_length=128,
            word_relations_filename='relation_data.json',
            word_relations_data_dir='/home/fm.vicente/explicit_commonsense/src/relation_attention',
            mask_heads=None
    ):
        super(MaskRelationsDataset, self).__init__(
            data,
            tokenizer=tokenizer,
            extra_answer_threshold=extra_answer_threshold,
            device=device,
            max_length=max_length,
            word_relations_filename=word_relations_filename,
            word_relations_data_dir=word_relations_data_dir,
            mask_heads=mask_heads
        )
