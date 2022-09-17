
#############################
#   Imports
#############################

# Python modules
import random
from typing import Tuple, Optional, List

# Remote modules

# Local modules
from .kg_base_wrapper import KGBaseHandler

from utils import read_json_file_2_dict

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class SwowHandler(KGBaseHandler):
    def __init__(self, store_dir='kgs_binding/swow'):
        super(SwowHandler, self).__init__()
        self.swow: dict = self.load_stored_data(store_dir=store_dir)

    def get_relation_types(self) -> List[str]:
        return ['related_to']

    def load_stored_data(self, filename='swow_knowledge.json', store_dir='kgs_binding/swow'):
        self.swow = read_json_file_2_dict(filename, store_dir)
        return self.swow

    def exists_relation_between(self, concept, other_concept):
        connections = self.swow.get(concept)
        if not connections:
            return False
        for connetion in connections:
            if connetion == other_concept:
                return True
        return False

    def does_concept_exist(self, concept):
        return self.swow.get(concept, None) is not None

    def relation_between(self, concept, other_concept) -> Tuple[Optional[str], Optional[str]]:
        exists_left_right = self.exists_relation_between(concept, other_concept)
        exists_right_left = self.exists_relation_between(other_concept, concept)
        relation = None
        if exists_left_right or exists_right_left:
            relation = 'related_to'
        return relation, relation

    def get_related_concepts(self, concept) -> Optional[List[str]]:
        return self.swow.get(concept, [])

    def simple_knowledge_prediction(self, knowledge):
        kw = list(knowledge)
        idx = random.randint(0, len(knowledge)-1) # 0-1-2
        kw[idx] = '<mask>'
        textual_knowledge_input = f'{kw[0]} {kw[1]} {kw[2]}'
        label = f'{knowledge[0]} {knowledge[1]} {knowledge[2]}'
        return f'{textual_knowledge_input},{label}\n', label

    def create_mask_knowledge_for_model(self):
        with open(f'bart_input/swow_bart.txt', 'w') as f:
            for subject, objects in self.swow.items():
                for obj in objects:
                    knowledge = (subject, 'is related to', obj)
                    w_kw, label = self.simple_knowledge_prediction(knowledge)
                    f.write(w_kw)

