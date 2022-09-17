
#############################
#   Imports
#############################

# Python modules

# Remote modules

# Local modules
from .relation_utils import clean_relations
from .DatasetGeneral import DatasetGeneral
from .datasets_model_handling import DatasetRelationsParsingUtils

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class RelationsDataset(DatasetGeneral):
    def __init__(
            self,
            datasets_parsing_utils:DatasetRelationsParsingUtils,
            relations_data=None,
            tokenizer=None,
            device=None,
            max_length=128,
            clear_common_wds = True
    ):
        print('initiated dataset')
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.device = device
        print('Loading dataset...')
        self.data = datasets_parsing_utils.get_data()
        self.phrases_relations = self.prepare_relations_data(datasets_parsing_utils, relations_data, clear_common_wds)
        print('self.data[0]:', self.data[0])
        print('self.phrases_relations[0]:', self.phrases_relations[0])
        print('tokenizing dataset')
        self.tokenized_data = self.tokenize_data()

    def prepare_relations_data(self, datasets_parsing_utils, relations_data, clear_common_wds):
        if relations_data is None:
            input_data = [data_unit.get('input_data') for data_unit in self.data]
            word_relations = datasets_parsing_utils.create_dataset_relations_mapping(input_data, clear_common_wds=clear_common_wds)
        else:
            word_relations = clean_relations(relations_data)
        return word_relations

    def make_example(self, idx) -> dict:
        question_sample, answer_sample, dataset_idx = self.get_sample(idx)
        #print('dataset_idx: ', dataset_idx)
        #print('len(self.word_relations): ', len(self.word_relations))
        #print('len(self.data): ', len(self.data))
        commonsense_relation = self.phrases_relations[dataset_idx]
        #print('commonsense_relation:', commonsense_relation)
        #print(question_sample, commonsense_relation)
        source = self.tokenizer(question_sample, padding='max_length',
                                truncation='longest_first',  max_length=self.max_length,
                                return_tensors="pt", return_offsets_mapping=True,
                                input_commonsense_relations=commonsense_relation,
                                )
        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer(answer_sample, padding='max_length',
                                    truncation='longest_first', max_length=self.max_length, #Todo
                                    input_commonsense_relations=None, return_offsets_mapping=True,
                                    return_tensors="pt")

        source_input_relations = source["input_commonsense_relations"].squeeze()
        data_example = self.transform_tokenization(source, target)
        data_example['input_commonsense_relations'] = source_input_relations
        return data_example


class MaskRelationsDataset(RelationsDataset):
    def __init__(self,
        datasets_parsing_utils:DatasetRelationsParsingUtils,
        relations_data=None,
        tokenizer=None,
        device=None,
        max_length=128,
        clear_common_wds=True
    ):
        super(MaskRelationsDataset, self).__init__(
            datasets_parsing_utils=datasets_parsing_utils,
            relations_data=relations_data,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            clear_common_wds=clear_common_wds
        )
