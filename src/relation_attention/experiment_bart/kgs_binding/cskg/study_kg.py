#############################
#   Imports
#############################

# Python modules
from tqdm import tqdm
import csv
# Remote modules

# Local modules

#############################
#   Constants
#############################

BASE_PATH = '/Users/mrvicente'

#############################
#   Stuff
#############################

def transverse_graph(cskg_path='./output/cskg_connected.kgtk'):
        knowledge_so_far = {}
        with open(cskg_path, 'r') as cskg_file:
            with open('./cskg_kb_2.txt', 'w') as cskg_writer:
                cskg_reader = csv.reader(cskg_file, delimiter='\t')
                cskg_header = next(cskg_reader, None)
                for line in tqdm(cskg_reader):
                    node1_label, relation, node2_label = line[4], line[6], line[5]
                    kb_type = line[6]
                    print()

if __name__ == '__main__':
    transverse_graph()