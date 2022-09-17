
#############################
#   Imports
#############################

# Python modules
from collections import deque
from collections import defaultdict
from typing import List, Dict, Optional
from ast import literal_eval
from random import sample

# Remote modules
#from tqdm.auto import tqdm

# Local modules
from .kg_base_wrapper import KGBaseHandler
from .swow_handler import SwowHandler
from utils import   (
    read_json_file_2_dict,
    Data_Type,
)
from .parsing_utils import ParsingUtils

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class RelationsMapperBuilder:
    def __init__(self,  knowledge: KGBaseHandler,
                        filename: Optional[str] = None,
                        file_dir: Optional[str] = None,
                        datatype: Optional[Data_Type] = None,
                        tok_sep:str = '</s>',
                        use_extra_relations=True):
        self.tok_sep = tok_sep
        self.knowledge = knowledge
        self.swow_knowledge = SwowHandler()
        self.use_extra_relations = use_extra_relations
        if filename and file_dir and datatype:
            full_context = self.load_data(filename, file_dir)
            self.relevant_context = self.fetch_relevant_context_from_data(data=full_context, datatype=datatype)

    def load_data(self, filename='commongen_qa_final.json', store_dir='./'):
        data = read_json_file_2_dict(filename=filename, store_dir=store_dir)
        print('data[0]:', data[0])
        return data

    def fetch_relevant_context_from_data(self, data: List[Dict], datatype:Data_Type = Data_Type.COMMONGEN_QA):
        if datatype == Data_Type.COMMONGEN_QA:
            model_input = [data_unit.get('title').lower() for data_unit in data]
        elif datatype in [Data_Type.ELI5, Data_Type.STACK_EXCHANGE]:
            model_input = [data_unit.get('question').lower() for data_unit in data]
        elif datatype in [Data_Type.COMMONSENSE_QA]:
            #questions = [data_unit.get('question').lower() for data_unit in data]
            #model_input = datasets_parsing_utils.compose_commonsenseqa_data(data)
            model_input = [data_unit.get('input_data') for data_unit in data]
        elif datatype in [Data_Type.COMMONGEN]:
            #questions = [data_unit.get('input_data').lower() for data_unit in data]
            #model_input = datasets_parsing_utils.compose_commongen_data(data)
            model_input = [data_unit.get('input_data') for data_unit in data]
        else:
            model_input = []
        return model_input

    def get_kg_concepts_from_context(self, context=None, clear_common_wds=False):
        if not context:
            context = self.relevant_context
        context_words = []
        for q_id, question in enumerate(context):
            simple_question = ParsingUtils.remove_pontuation(question)
            n_grams = ParsingUtils.n_grams_n_words_extractor(simple_question)
            words = self.relevant_entities_extractor(n_grams)
            if clear_common_wds:
                words = ParsingUtils.clear_common_words(words)
            simple_words = [word[0] for word in words]
            context_words.append(simple_words)
        return context_words

    def obtain_concept_neighbours(self, context_concepts:List[str], n_neighbours = 20):
        """
        Use swow to get connected concepts, but then refer back to conceptnet for rich relations
        """
        neighbours = []
        for concept in context_concepts:
            external_neighbour_concepts = self.swow_knowledge.get_related_concepts(concept)
            relevant_concepts = external_neighbour_concepts
            #local_neighbour_concepts = self.knowledge.get_related_concepts(concept)
            #relevant_concepts = [ext_concept for ext_concept in external_neighbour_concepts if ext_concept in local_neighbour_concepts]
            neighbours.extend(relevant_concepts)
        n_neighbours = min(n_neighbours, len(neighbours))
        some_neighbours = sample(neighbours, n_neighbours)
        #print('context_concepts:', context_concepts)
        #print('some_neighbours:', some_neighbours)
        return some_neighbours


    def get_relations_mapping_complex(self, context=None, clear_common_wds=False):
        if not context:
            context = self.relevant_context
        relations_info = deque()
        for q_id, question in enumerate(context):
            simple_question = ParsingUtils.remove_pontuation(question)
            n_grams = ParsingUtils.n_grams_n_words_extractor(simple_question)
            words = self.relevant_entities_extractor(n_grams)
            if clear_common_wds:
                words = ParsingUtils.clear_common_words(words)
            #print(f'question: {question}')
            #print(f'words: {words}')
            relation_context_between_words = defaultdict(dict)
            known_tokens = set()
            for token_i, (first_word_token, first_word_range) in enumerate(words[:-1]):
                known_tokens.add(first_word_token)
                first_word_range_str = str(first_word_range)
                # normalize
                first_word_phrase_normalized = self.knowledge.normalize_nouns(first_word_token)
                for (second_word_token, second_word_range) in [w for w in words[token_i + 1:] if w not in known_tokens]:
                    second_word_range_str = str(second_word_range)
                    second_word_phrase_normalized = self.knowledge.normalize_nouns(second_word_token)
                    left_2_right, right_2_left = self.knowledge.relation_between(first_word_phrase_normalized, second_word_phrase_normalized)
                    #print(first_word_token, second_word_token, left_2_right, right_2_left)
                    if left_2_right:
                        relation_context_between_words[first_word_range_str][second_word_range_str] = left_2_right
                    if right_2_left:
                        relation_context_between_words[second_word_range_str][first_word_range_str] = right_2_left
            relations_info.append(dict(relation_context_between_words))
        return list(relations_info)

    def get_concepts_from_context(self, context=None, clear_common_wds=False,alignment=0):
        relations_info = self.get_relations_mapping_complex(context=[context], clear_common_wds=clear_common_wds)
        words = []
        #print('relations_info here:', relations_info)
        for rels in relations_info:
            for coords, v in rels.items():
                coords_tuple = literal_eval(coords)
                i,j = coords_tuple
                words.append(context[i+alignment:j+alignment])
                for coords_other, rel in v.items():
                    coords_other_tuple = literal_eval(coords_other)
                    i_other, j_other = coords_other_tuple
                    words.append(context[i_other+alignment: j_other+alignment])
        returning_words = list(set(words))
        #print('returning_words:', returning_words)
        return returning_words

    def relevant_entities_extractor(self, n_grams_n_words, verbose_output=True):
        non_overlapping_knowledge = {}
        # print(n_grams_n_words)
        for concept, (idx_start, idx_end) in n_grams_n_words:
            normalized_concept = self.knowledge.normalize_nouns(concept)
            exists = self.knowledge.does_concept_exist(normalized_concept)
            #print('exists: ', concept, normalized_concept, exists)
            if exists and idx_start not in non_overlapping_knowledge and \
                idx_end not in non_overlapping_knowledge:
                non_overlapping_knowledge[idx_start] = (concept, idx_start, idx_end, 'start_idx')
                non_overlapping_knowledge[idx_end] = (concept, idx_end, idx_end, 'end_idx')
        if verbose_output:
            return [(value[0], (value[1], value[2])) for k, value in sorted(non_overlapping_knowledge.items()) if value[-1] == 'start_idx']
        else:
            return [value[0] for k, value in sorted(non_overlapping_knowledge.items()) if value[-1] == 'start_idx']
