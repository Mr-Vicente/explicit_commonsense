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

class CSKGHandler(KGBaseHandler):
    def __init__(self):
        super(CSKGHandler, self).__init__()
        self.kg = read_json_file_2_dict('cskg_node_2_node_relations.json', store_dir='cskg')
        self.kg_nodes = read_json_file_2_dict('cskg_node_2_nodes.json', store_dir='cskg')

    def get_relation_types(self) -> List[str]:
        #relations = list(set([y for x in list(self.kg.values()) for y in x]))
        relations = read_json_file_2_dict('relations.json', store_dir='cskg')
        return relations

    def exists_relation_between(self, concept, other_concept) -> bool:
        left_2_right, right_2_left = self.relation_between(concept, other_concept)
        return left_2_right is not None or right_2_left is not None

    def relation_between(self, concept, other_concept) -> Tuple[Optional[str], Optional[str]]:
        left_2_right_txt = f'{concept}|{other_concept}'
        right_2_left_txt = f'{other_concept}|{concept}'
        left_2_right_relations = self.kg.get(left_2_right_txt, None)
        right_2_left_relations = self.kg.get(right_2_left_txt, None)
        left_2_right_relation, right_2_left_relation = None, None
        if left_2_right_relations:
            left_2_right_relation = self.ignore_less_relevant_connection(left_2_right_relations)
        if right_2_left_relations:
            right_2_left_relation = self.ignore_less_relevant_connection(right_2_left_relations)
        return left_2_right_relation, right_2_left_relation

    def does_concept_exist(self, concept) -> bool:
        return self.kg_nodes.get(concept, None) is not None

