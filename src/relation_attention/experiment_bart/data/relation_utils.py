
#############################
#   Imports
#############################

# Python modules
from collections import deque
from ast import literal_eval

# Remote modules

# Local modules

#############################
#   Constants
#############################

##########################################################
#   Helper functions for Relations in dict format
##########################################################
import torch


def clean_relations(word_relations):
    new_relations = deque()
    for r in word_relations:
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

##########################################################
#   Helper functions for Relations in Matrix format
##########################################################

def relation_binary_2d_to_1d(relations_binary_mask, dim=1):
    relations_binary_mask = relations_binary_mask.sum(dim=dim)
    relations_binary_mask[relations_binary_mask > 1] = 1
    return relations_binary_mask

def tokens_with_relations(relations_binary_mask):
    relations_binary_mask_dim1 = relations_binary_mask.sum(dim=0)
    relations_binary_mask_dim2 = relations_binary_mask.sum(dim=1)
    tokens_with_rels = relations_binary_mask_dim1 + relations_binary_mask_dim2
    tokens_with_rels[tokens_with_rels > 1] = 1
    mask_rels = torch.tensor(tokens_with_rels, dtype=torch.bool)
    return mask_rels

""" Mutual
def tokens_with_relations(relations_binary_mask):
    relations_binary_mask_dim1 = relations_binary_mask.sum(dim=0)
    relations_binary_mask_dim2 = relations_binary_mask.sum(dim=1)
    print('relations_binary_mask_dim1:', relations_binary_mask_dim1)
    print('relations_binary_mask_dim2:', relations_binary_mask_dim2)
    relations_binary_mask_dim1[relations_binary_mask_dim1 > 1] = 1
    relations_binary_mask_dim2[relations_binary_mask_dim2 > 1] = 1
    tokens_with_rels = relations_binary_mask_dim1 # relations_binary_mask_dim2
    print('s_rels:', tokens_with_rels)
    return tokens_with_rels
"""
