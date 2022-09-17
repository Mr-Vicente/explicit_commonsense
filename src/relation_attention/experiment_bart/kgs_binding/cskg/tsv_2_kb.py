#############################
#   Imports
#############################

# Python modules
from ast import literal_eval


# Remote modules

# Local modules
import csv
from tqdm import tqdm

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

def parse_knowledge(cskg_writer, relation, node1, node2, knowledge_so_far):
    relation_vec = relation.split('|')
    for relation_unit in relation_vec:
        for node1 in parse_node(node1):
            for node2 in parse_node(node2):
                relation_unit, node1, node2 = clean_str(relation_unit), clean_str(node1), clean_str(node2)
                knowlegde_str = f'{relation_unit}("{node1}","{node2}")'
                k = knowledge_so_far.get(knowlegde_str, None)
                if not k:
                    cskg_writer.write(f'{knowlegde_str}\n')
                    knowledge_so_far[knowlegde_str] = knowlegde_str


def parse_cskg(cskg_path='./output/cskg_connected.kgtk'):
    knowledge_so_far = {}
    with open(cskg_path, 'r') as cskg_file:
        with open('./cskg_kb_2.txt', 'w') as cskg_writer:
            cskg_reader = csv.reader(cskg_file, delimiter='\t')
            cskg_header = next(cskg_reader, None)
            for line in tqdm(cskg_reader):
                node1_label, relation, node2_label = line[4], line[6], line[5]
                #print(node1_label, ' | ', relation, ' | ', node2_label)
                relation_vec = relation.split(' ')
                relation_str = '_'.join(relation_vec)
                #if 'enjoys_h' in relation_str:
                #    print(relation_str, "-------", node1_label, relation, node2_label)
                parse_knowledge(cskg_writer, relation_str, node1_label, node2_label, knowledge_so_far)
                #knowlegde_str = f'{relation_str}("{node1_label}","{node2_label}")'
                #cskg_writer.write(f'{knowlegde_str}\n')
                #print(knowlegde_str)

if __name__ == '__main__':
    parse_cskg()
