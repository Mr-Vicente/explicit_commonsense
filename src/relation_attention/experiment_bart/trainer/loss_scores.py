#############################
#   Imports
#############################

# Python modules
import math
# Remote modules
import torch

# Local modules

#############################
#   Scoring functions
#############################

def score_based_on_matches(relevant_ids, pred_ids):
    """
    print('<<<<<<<<<<<<>>>>>>>>>>>')
    print('relevant_ids:',relevant_ids)
    print('pred_ids:', pred_ids)
    print('<<<<<<<<<<<<>>>>>>>>>>>')
    """
    counter = 1
    if len(relevant_ids) == 0:
        return 1
    for pred_id in pred_ids:
        if pred_id in relevant_ids:
            counter += 1
    n_relevant_ids = len(relevant_ids)
    f = len(pred_ids)/(1 + math.exp(2*(counter-(0.7 *n_relevant_ids) ) ) )
    # len(pred_ids)/(counter**2)
    return f

def score_on_concept_richness(gen_concepts, label_concepts, seq_len):
    n_label_concepts = len(label_concepts)
    concept_counter=0
    for gen_concept in gen_concepts:
        if gen_concept in label_concepts:
            concept_counter+=1
    f = seq_len / (1 + math.exp(2*(concept_counter-(0.7 *n_label_concepts) ) ) )
    return f
#    f = len(pred_ids) * ( math.exp(-counter/(0.2 + (0.1*n_relevant_ids))) / ((0.3*math.exp(-counter/(0.2 + (0.1*n_relevant_ids))))+0.05))

def score_based_on_amount_of_relations(rels_info, original_rels_amount=2):
    counter = 1
    if original_rels_amount == 0:
        original_rels_amount = 2
    #print('rels_info:', rels_info)
    for concept, outgoing_rels in rels_info.items():
        counter += 3*len(outgoing_rels.values())
    return original_rels_amount/counter

def old_score_based_on_lack_of_relational_concepts(concepts, concepts_without_rels):
    n_concepts = len(concepts)
    n_concepts = 1 if n_concepts == 0 else n_concepts
    n_concepts_without_rels = len(concepts_without_rels)
    return n_concepts_without_rels**2/n_concepts

def score_based_on_lack_of_relational_concepts(concepts, concepts_without_rels, amount_of_relations, seq_len=32):
    n_concepts = len(concepts)
    n_concepts = 1 if n_concepts == 0 else n_concepts
    n_concepts_without_rels = len(concepts_without_rels)
    n_concepts_with_rels = n_concepts-n_concepts_without_rels
    n_concepts_with_rels = 1 if n_concepts_with_rels == 0 else n_concepts_with_rels
    #f = seq_len * ( math.exp(-n_concepts_with_rels/(0.2 + (0.1*amount_of_relations))) / ((0.3*math.exp(-n_concepts_with_rels/(0.2 + (0.1*amount_of_relations))))+0.05))
    f = seq_len / (1 + math.exp(2 * (n_concepts_with_rels - (0.7 * amount_of_relations))))
    # 32/(n_concepts_with_rels**2)
    return f