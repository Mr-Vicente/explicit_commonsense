
#############################
#   Imports
#############################

# Python modules
import csv
from collections import defaultdict
import re

# Remote modules
from tqdm import tqdm

# Local modules
from src.utils import write_dict_2_json_file

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

def uri_to_text(uri):
    text_vec = uri.split('/')
    _language, text = text_vec[2:4]
    clean_text = text.replace('_', ' ')
    return clean_text

def clear_relation(relation):
    cleaned_relation = relation.split('/')[-1]
    cleaned_relation_vec = re.findall('[A-Z][^A-Z]*', cleaned_relation)
    cleaned_relation_str = '_'.join(cleaned_relation_vec).lower()
    return cleaned_relation_str

def create_simple_nouns_file():
    with open('/Users/mrvicente/Downloads/conceptnet-assertions-5.7.0.csv') as f:
        reader = csv.reader(f, delimiter='\t')
        d_english = defaultdict(str)
        for idx, line in tqdm(enumerate(reader)):
            relation, subject, object = line[1:4]
            is_english = '/en/' in subject
            if is_english:
                #print(idx, subject, relation, object)
                subject_text = uri_to_text(subject)
                object_text = uri_to_text(object)
                #print(subject_text,object_text)
                if object_text not in d_english[subject_text]:
                    d_english[subject_text] = ''
        write_dict_2_json_file(d_english, 'conceptnet_english_nouns_simple.json')

def create_noun_2_noun_file():
    with open('/Users/mrvicente/Downloads/conceptnet-assertions-5.7.0.csv') as f:
        reader = csv.reader(f, delimiter='\t')
        d_english = defaultdict(list)
        for idx, line in tqdm(enumerate(reader)):
            relation, subject, object = line[1:4]
            sub_is_english = '/en/' in subject
            obj_is_english = '/en/' in object
            if sub_is_english and obj_is_english:
                clean_relation = clear_relation(relation)
                #print(clean_relation)
                #print(idx, subject, relation, object)
                subject_text = uri_to_text(subject)
                object_text = uri_to_text(object)
                #print(subject_text,object_text)
                d_english[f'{subject_text}|{object_text}'].append(clean_relation)
        write_dict_2_json_file(d_english, 'conceptnet_english_noun_2_noun_relations.json')


if __name__ == '__main__':
    create_noun_2_noun_file()