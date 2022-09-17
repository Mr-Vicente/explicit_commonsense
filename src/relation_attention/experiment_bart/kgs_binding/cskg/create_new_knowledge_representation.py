#############################
#   Imports
#############################

# Python modules
import csv
from tqdm import tqdm
from ast import literal_eval

# Remote modules

# Local modules
from src.utils import write_dict_2_json_file

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

def clean_str(n):
    try:
        x = literal_eval(n)
    except Exception as v_e:
        x = n
    return x

def parse_node(node):
    if type(node) is not str:
        node = str(node)
    node_vec = node.split('|')
    for node_unit in node_vec:
        yield node_unit

def update_entry_on_knowledge(kg, key, value):
    connections = kg.get(key, [])
    connections.append(value)
    kg[key] = connections

def parse_knowledge(relation, node1, node2, knowledge_so_far, graph_representation, graph_rel_representation):
    relation_vec = relation.split('|')
    for relation_unit in relation_vec:
        for node1 in parse_node(node1):
            for node2 in parse_node(node2):
                relation_unit, node1, node2 = clean_str(relation_unit), clean_str(node1), clean_str(node2)
                knowlegde_str = f'{relation_unit}("{node1}","{node2}")'
                k = knowledge_so_far.get(knowlegde_str, None)
                if not k:
                    knowledge_so_far[knowlegde_str] = knowlegde_str
                    update_entry_on_knowledge(graph_representation, node1, node2)
                    update_entry_on_knowledge(graph_rel_representation, f'{node1}|{node2}', relation_unit)


def parse_cskg(cskg_path='./output/cskg_connected.kgtk'):
    knowledge_so_far = {}
    graph_representation = {}
    graph_rel_representation = {}
    with open(cskg_path, 'r') as cskg_file:
        cskg_reader = csv.reader(cskg_file, delimiter='\t')
        cskg_header = next(cskg_reader, None)
        for line in tqdm(cskg_reader):
            node1_label, relation, node2_label = line[4], line[6], line[5]
            relation_vec = relation.split(' ')
            relation_str = '_'.join(relation_vec)
            parse_knowledge(relation_str, node1_label, node2_label, knowledge_so_far,
                            graph_representation, graph_rel_representation)
    write_dict_2_json_file(json_object=graph_representation,
                           filename='cskg_node_2_nodes.json',
                           store_dir='')
    write_dict_2_json_file(json_object=graph_rel_representation,
                           filename='cskg_node_2_node_relations.json',
                           store_dir='')

if __name__ == '__main__':
    parse_cskg()
