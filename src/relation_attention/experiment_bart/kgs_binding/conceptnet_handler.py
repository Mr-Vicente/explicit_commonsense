#############################
#   Imports
#############################

# Python modules
from typing import Tuple, Optional, List
# Remote modules

# Local modules
from .kg_base_wrapper import KGBaseHandler
from utils import read_json_file_2_dict

#############################
#   Constants
#############################

#############################
#   Handler
#############################

class ConceptNetHandler(KGBaseHandler):
    def __init__(self, database="/home/fm.vicente/thesis_datasets/Knowledge_Graphs/conceptnet.db"):
        super(ConceptNetHandler, self).__init__()
        #self.conceptnet = Concept_Net(database=database)
        _store_dir = 'kgs_binding/conceptnet'
        #_store_dir = './conceptnet'
        self.conceptnet_concepts = read_json_file_2_dict('conceptnet_english_nouns_simple.json', store_dir=_store_dir)
        self.relations_concepts = read_json_file_2_dict('conceptnet_english_noun_2_noun_relations.json', store_dir=_store_dir)
        self.concept_2_concepts = read_json_file_2_dict('conceptnet_english_nouns.json', store_dir=_store_dir)

    def get_relation_types(self) -> List[str]:
        relation_names = ['derived_from', 'dbpedia/genre', 'antonym', 'desires', 'related_to', 'part_of',
                          'used_for', 'causes_desire', 'capable_of', 'has_context', 'entails', 'motivated_by_goal',
                          'causes', 'distinct_from', 'made_of', 'synonym', 'at_location', 'manner_of',
                          'etymologically_related_to', 'has_subevent', 'has_prerequisite', 'not_desires',
                          'has_property', 'similar_to', 'is_a', 'has_a', 'form_of', 'has_first_subevent', 'has_last_subevent',
                          'obstructed_by', 'created_by', 'symbol_of', 'defined_as', 'located_near', 'etymologically_derived_from',
                          'receives_action', 'external_url']
        updated_relation_names = ['not_has_property', 'not_desires', 'external_u_r_l', 'created_by',
                          'not_capable_of', 'antonym', 'has_first_subevent', 'located_near',
                          'desires', 'has_prerequisite', 'has_last_subevent', 'synonym', 'is_a',
                          'manner_of', 'has_a', 'motivated_by_goal', 'instance_of',
                          'etymologically_derived_from', 'capable_of', 'for', 'at_location',
                          'has_subevent', 'causes', 'has_context', 'symbol_of', 'derived_from',
                          'made_of', 'causes_desire', 'has_property', 'similar_to', 'used_for', 'by',
                          'entails', 'form_of', 'receives_action', 'distinct_from', 'related_to',
                          'part_of', 'defined_as', 'etymologically_related_to']
        return updated_relation_names

    def exists_relation_between(self, concept, other_concept) -> bool:
        left_2_right, right_2_left = self.relation_between(concept, other_concept)
        return left_2_right is not None or right_2_left is not None

    def relation_between(self, concept, other_concept) -> Tuple[Optional[str], Optional[str]]:
        left_2_right_txt = f'{concept}|{other_concept}'
        right_2_left_txt = f'{other_concept}|{concept}'
        left_2_right_relations = self.relations_concepts.get(left_2_right_txt, None)
        right_2_left_relations = self.relations_concepts.get(right_2_left_txt, None)
        left_2_right_relation, right_2_left_relation = None, None
        if left_2_right_relations:
            left_2_right_relation = self.ignore_less_relevant_connection(left_2_right_relations)
        if right_2_left_relations:
            right_2_left_relation = self.ignore_less_relevant_connection(right_2_left_relations)
        return left_2_right_relation, right_2_left_relation

    def get_related_concepts(self, concept) -> Optional[List[str]]:
        return self.concept_2_concepts.get(concept, [])

    def does_concept_exist(self, concept) -> bool:
        return concept in self.conceptnet_concepts # old: self.conceptnet.exists_concept(concept)
