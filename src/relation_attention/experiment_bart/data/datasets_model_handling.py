#############################
#   Imports
#############################

# Python modules
from typing import (
    List,
    Dict
)
from collections import deque

# Remote modules

# Local modules
from utils import (
    Data_Type
)
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder
from .relation_utils import clean_relations

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

SEP_TOKEN = '<s>'

class DatasetParsingUtils:
    def __init__(self,
                 dataset_type:Data_Type,
                 data:List[Dict],
                 use_context:bool=True,
                 sep_token:str=SEP_TOKEN
                 ):
        self.dataset_type = dataset_type
        self.data = data
        self.sep_token = sep_token
        self.use_context = use_context
        self.idx = 0

    def get_commonsenseqa_labels(self, data_unit):
        #data_unit.get('answers', {}).get('text', [])[0].lower()
        labels_vec = data_unit.get('answers', {}).get('text', [])
        labels_str = ';'.join(labels_vec).lower() #join_str_first(' o:', labels_vec).lower()
        return labels_str
    def get_commongen_labels(self, data_unit):
        return data_unit.get('labels').lower()
    def get_commonsenseqa_context(self, data_unit):
        list_of_choices = data_unit.get('context')
        list_of_choices_str = ';'.join(list_of_choices).lower() #join_str_first(' o:', list_of_choices).lower()
        return list_of_choices_str
    def get_eli5_alike_labels(self, data_unit, extra_answer_threshold=1):
        label_text = None
        for j, (answer, sc) in enumerate(zip(data_unit["answers"]["text"], data_unit["answers"]["score"])):
            if j == 0 and int(sc) >= extra_answer_threshold and answer:
                label_text=answer.lower()
        return label_text

    def build_model_input(self, input_data, output_data, idx=None):
        if idx is None:
            idx = self.idx
            self.idx += 1
        return {
            'input_data': input_data,
            'labels_data': output_data,
            'idx': idx
        }

    def get_data(self):
        if self.dataset_type == Data_Type.ELI5:
            return self.compose_eli5_alike_data(self.data)
        elif self.dataset_type == Data_Type.COMMONGEN:
            return self.compose_commongen_data(self.data)
        elif self.dataset_type == Data_Type.COMMONSENSE_QA:
            return self.compose_commonsenseqa_data(self.data)
        else:
            raise NotImplementedError()

    def compose_eli5_alike_data(self, data):
        temp_data = deque()
        for data_unit in data:
            phrase = data_unit.get('question').lower()
            if self.use_context:
                context = data_unit.get('context').lower()
                model_input_phrase = f'{phrase}{self.sep_token}{context}'
            else:
                model_input_phrase = phrase
            output = self.get_eli5_alike_labels(data_unit)
            if output:
                idx = data_unit.get('idx')
                model_input = self.build_model_input(model_input_phrase, output, idx)
                temp_data.append(model_input)
        return list(temp_data)

    def compose_commongen_data(self, data):
        temp_data = deque()
        for data_unit in data:
            phrase = data_unit.get('input_data').lower()
            output = self.get_commongen_labels(data_unit)
            model_input_phrase = f'{phrase}'
            model_input = self.build_model_input(model_input_phrase, output)
            temp_data.append(model_input)
        return list(temp_data)

    def compose_commonsenseqa_data(self, data):
        temp_data = deque()
        for data_unit in data:
            phrase = data_unit.get('question').lower()
            if self.use_context:
                context = self.get_commonsenseqa_context(data_unit)
                model_input_phrase = f'{phrase}{self.sep_token}{context}'
            else:
                model_input_phrase = phrase
            output = self.get_commonsenseqa_labels(data_unit)
            model_input = self.build_model_input(model_input_phrase, output)
            temp_data.append(model_input)
        return list(temp_data)


class DatasetRelationsParsingUtils(DatasetParsingUtils):
    def __init__(self,
                 dataset_type: Data_Type,
                 data: List[Dict],
                 relations_mapper: RelationsMapperBuilder,
                 use_context: bool = True,
                 sep_token: str = SEP_TOKEN,
                 use_extra_relations: bool = True,
                 ):
        super(DatasetRelationsParsingUtils, self).__init__(
            dataset_type=dataset_type,
            data=data,
            use_context=use_context,
            sep_token=sep_token
        )
        self.relations_mapper = relations_mapper
        self.use_extra_relations = use_extra_relations

    def create_dataset_relations_mapping(self, data, clear_common_wds=True):
        phrases_relations_str = self.relations_mapper.get_relations_mapping_complex(context=data, clear_common_wds=clear_common_wds)
        phrases_relations = clean_relations(phrases_relations_str)
        return phrases_relations

    def fetch_neighbour_concepts_from_phrase(self, phrase):
        concepts = self.relations_mapper.get_kg_concepts_from_context([phrase], clear_common_wds=True)[0]
        other_concepts = self.relations_mapper.obtain_concept_neighbours(concepts)
        other_concepts_str = ' '.join(other_concepts)
        return other_concepts_str

    def compose_eli5_alike_data(self, data):
        if not self.use_extra_relations:
            return super().compose_commonsenseqa_data(data)
        temp_data = deque()
        for data_unit in data:
            phrase = data_unit.get('question').lower()
            other_concepts_str = self.fetch_neighbour_concepts_from_phrase(phrase)
            model_input_phrase = phrase
            if self.use_context:
                context = data_unit.get('context').lower()
                model_input_phrase = f'{phrase}{self.sep_token}{context}'
            if self.use_extra_relations:
                model_input_phrase = f'{model_input_phrase}{self.sep_token}{other_concepts_str}'
            output = self.get_eli5_alike_labels(data_unit)
            if output:
                idx = data_unit.get('idx')
                model_input = self.build_model_input(model_input_phrase, output, idx)
                temp_data.append(model_input)
        return list(temp_data)

    def compose_commongen_data(self, data):
        if not self.use_extra_relations:
            return super().compose_commongen_data(data)
        temp_data = deque()
        for data_unit in data:
            phrase = data_unit.get('input_data').lower()
            other_concepts_str = self.fetch_neighbour_concepts_from_phrase(phrase)
            model_input_phrase = phrase
            if self.use_extra_relations:
                model_input_phrase = f'{model_input_phrase}{self.sep_token}{other_concepts_str}'
            output = self.get_commongen_labels(data_unit)
            model_input = self.build_model_input(model_input_phrase, output)
            temp_data.append(model_input)
        return list(temp_data)

    def compose_commonsenseqa_data(self, data):
        if not self.use_extra_relations:
            return super().compose_commonsenseqa_data(data)
        temp_data = deque()
        for data_unit in data:
            phrase = data_unit.get('question').lower()
            other_concepts_str = self.fetch_neighbour_concepts_from_phrase(phrase)
            if self.use_context:
                context = self.get_commonsenseqa_context(data_unit)
                model_input_phrase = f'{phrase}{self.sep_token}{context}{self.sep_token}{other_concepts_str}'
            else:
                model_input_phrase = f'{phrase}{self.sep_token}{other_concepts_str}'
            output = self.get_commonsenseqa_labels(data_unit)
            model_input = self.build_model_input(model_input_phrase, output)
            temp_data.append(model_input)
        return list(temp_data)

