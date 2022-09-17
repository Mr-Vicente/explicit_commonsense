#############################
#   Imports
#############################

# Python modules
import csv
from tqdm import tqdm
from ast import literal_eval
import random
from copy import deepcopy

# Remote modules

# Local modules
from src.utils import (
    write_dict_2_json_file,
    read_json_file_2_dict
)

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class CSKG_Parser:
    def __init__(self, kg_path='./output/cskg_connected.kgtk'):
        self.kg_path = kg_path

    def _clean_str(self, n):
        try:
            x = literal_eval(n)
        except Exception as v_e:
            x = n
        return x

    def _parse_node(self, node):
        if type(node) is not str:
            node = str(node)
        node_vec = node.split('|')
        for node_unit in node_vec:
            yield node_unit

    def parse_knowledge(self, cskg_writer, relation, node1, node2, knowledge_so_far):
        relation_vec = relation.split('|')
        for relation_unit in relation_vec:
            for node1 in self._parse_node(node1):
                for node2 in self._parse_node(node2):
                    relation_unit, node1, node2 = self._clean_str(relation_unit), self._clean_str(node1), self._clean_str(node2)
                    knowlegde_str = f'{relation_unit}("{node1}","{node2}")'
                    k = knowledge_so_far.get(knowlegde_str, None)
                    if not k:
                        cskg_writer.write(f'{knowlegde_str}\n')
                        knowledge_so_far[knowlegde_str] = knowlegde_str

    def parse_cskg(self, cskg_path=None):
        if not cskg_path:
            cskg_path = self.kg_path
        knowledge_so_far = {}
        with open(cskg_path, 'r') as cskg_file:
            with open('./cskg_kb_2.txt', 'w') as cskg_writer:
                cskg_reader = csv.reader(cskg_file, delimiter='\t')
                _cskg_header = next(cskg_reader, None)
                for line in tqdm(cskg_reader):
                    node1_label, relation, node2_label = line[4], line[6], line[5]
                    relation_vec = relation.split(' ')
                    relation_str = '_'.join(relation_vec)
                    self.parse_knowledge(cskg_writer, relation_str, node1_label, node2_label, knowledge_so_far)

    def gen_candidate(self, kg_structured, kg_relations, candidates, visited, starting_node):
        node2_candidates = kg_structured.get(starting_node, [None])
        node2 = random.choice(node2_candidates)
        if not node2 or (node2 in visited and len(candidates) <= 2):
            return None
        relations = kg_relations.get(f'{starting_node}|{node2}')
        relation = random.choice(relations)
        candidate = [starting_node, relation, node2]
        visited.append(starting_node)
        return candidate

    def hallucinate(self,
                    kg_structured,
                    kg_relations,
                    candidates,
                    visited,
                    starting_node,
                    max_hop_hallucinated=2
                    ):
        visited_hall = deepcopy(visited)
        candidates_hall = []
        for hop_state in range(1, max_hop_hallucinated + 1):
            candidate = self.gen_candidate(kg_structured, kg_relations, candidates, visited_hall, starting_node)
            if not candidate:
                candidates_hall = []
                break
            candidates_hall.append(candidate)
            node2 = candidate[-1]
            starting_node = node2
        return candidates_hall


    def random_walk_gen(self,
                        kg_structured_for_walking_path='./cskg_node_2_nodes.json',
                        kg_relations_for_walking_path='./cskg_node_2_node_relations.json',
                        max_hop_length=5,
                        dataset_entries=10000,
                        cskg_random_walk_dataset_path='./cskg_random_walk_dataset.json'
                        ):
        kg_structured = read_json_file_2_dict(kg_structured_for_walking_path)
        kg_relations = read_json_file_2_dict(kg_relations_for_walking_path)
        concepts = list(kg_structured.keys())
        dataset = []
        while len(dataset) <= dataset_entries:
            starting_node = random.choice(concepts)
            candidates = []
            visited = []
            max_hop = random.randint(3, max_hop_length)
            for hop_state in range(1, max_hop+1):
                candidate = self.gen_candidate(kg_structured, kg_relations, candidates, visited, starting_node)
                if not candidate:
                    candidates = []
                    break
                candidates.append(candidate)
                node2 = candidate[-1]
                starting_node = node2
                local_hallucinations = self.hallucinate(kg_structured, kg_relations, candidates, visited, starting_node)
                candidates.extend(local_hallucinations)
            global_hallucinations = self.hallucinate(kg_structured, kg_relations, candidates, visited, random.choice(concepts))
            candidates.extend(global_hallucinations)
            # hallucination time!
            if candidates:
                first_candidate = candidates[0]
                random.shuffle(candidates)
                last_node = starting_node
                start_node, start_relation = first_candidate[0], first_candidate[1]
                dataset_entry = {
                    'query': [start_node, start_relation, 'X'],
                    'candidates': [last_node],
                    'supports': candidates,
                    'answer': f'{last_node}'
                }
                dataset.append(dataset_entry)

        write_dict_2_json_file(json_object=dataset,
                               filename=cskg_random_walk_dataset_path,
                               store_dir='')

if __name__ == '__main__':
    cs_parser = CSKG_Parser()
    path = 'cskg_random_walk_dataset.json'
    # training data
    cs_parser.random_walk_gen(cskg_random_walk_dataset_path=f'./train_{path}')
    # validation data
    cs_parser.random_walk_gen(dataset_entries=2000, cskg_random_walk_dataset_path=f'./val_{path}')






