
#############################
#   Imports
#############################

# Python modules
from collections import deque
from ast import literal_eval

# Remote modules
from tqdm import tqdm

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

def clean_relations(word_relations):
    new_relations = deque()
    for r in tqdm(word_relations):
        rel = {}
        for r_key, r_value in r.items():
            normal_k = literal_eval(r_key)
            rel_d = {}
            for r_d_key, r_d_value in r_value.items():
                normal_d_k = literal_eval(r_d_key)
                rel_d[normal_d_k] = r_d_value
            rel[normal_k] = rel_d
        new_relations.append(rel)
    list_new_relations = list(new_relations)
    return list_new_relations


def embedd_buffer(buffer_sentences, sentences_embeddings):
    import numpy as np
    buffer_embeddings = None #self._get_embedding(buffer_sentences)
    buffer_sentences = []
    if len(sentences_embeddings) != 0:
        sentences_embeddings = np.concatenate([sentences_embeddings, buffer_embeddings])
    else:
        sentences_embeddings = buffer_embeddings
    return sentences_embeddings, buffer_sentences