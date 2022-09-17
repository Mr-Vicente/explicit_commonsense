
import numpy as np
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
)

from collections import deque
from ast import literal_eval

from utils import read_json_file_2_dict

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
    # print(list_new_relations)
    return list_new_relations

def m(tokenizer_outputs, input_relations):
    relation_kinds = ['derived_from', 'dbpedia/genre', 'antonym', 'desires', 'related_to', 'part_of',
                      'used_for', 'causes_desire', 'capable_of', 'has_context', 'entails', 'motivated_by_goal',
                      'causes', 'distinct_from', 'made_of', 'synonym', 'at_location', 'manner_of',
                      'etymologically_related_to', 'has_subevent', 'has_prerequisite', 'not_desires',
                      'has_property', 'similar_to', 'is_a', 'has_a']
    relational_kind_to_index = {t: i + 1 for i, t in enumerate(relation_kinds)}
    aux_input_relation_kinds = np.zeros(
        (len(tokenizer_outputs['input_ids']), len(tokenizer_outputs['input_ids'][0]),
         len(tokenizer_outputs['input_ids'][0])),
        dtype=np.int64
    )
    for batch_idx, (token_mappings, relations) in enumerate(zip(tokenizer_outputs['offset_mapping'], input_relations)):

        for word_i_span, word_relations in relations.items():
            word_i_token_ids = [
                token_idx for token_idx, token_span in enumerate(token_mappings)
                if max(0, min(token_span[1], word_i_span[1]) - max(token_span[0], word_i_span[0])) > 0
                # check for word/token overlaps
            ]
            for word_j_span, relation_kind in word_relations.items():
                for token_j_idx, token_span in enumerate(token_mappings):
                    if max(0, min(token_span[1], word_j_span[1]) - max(token_span[0], word_j_span[
                        0])) > 0:  # check for word/token overlaps
                        for token_i_idx in word_i_token_ids:
                            try:
                                aux_input_relation_kinds[batch_idx, token_i_idx, token_j_idx] = \
                                    relational_kind_to_index[relation_kind]

                            except IndexError:
                                raise IndexError(f"Could not find relation kind '{relation_kind}'")
        print("aux_input_relation_kinds: ", aux_input_relation_kinds)

if __name__ == '__main__':
    idx = 0
    word_relations = clean_relations(read_json_file_2_dict('relation_data.json')[:1])
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    question_sample = "What is happening when people seldomly have commonsense in a tube train?"
    #source = tokenizer(question_sample, padding='max_length', input_relations=word_relations[idx], return_offsets_mapping=True)
    source = tokenizer(question_sample, padding='max_length', return_offsets_mapping=True, return_tensors='pt')
    print("source['input_ids']:", source['input_ids'])
    print("source['word_ids']:", source['word_ids'])
    print("to normal:", tokenizer.convert_ids_to_tokens(source['input_ids'][0]))
    print("to normal:", tokenizer.batch_decode(source['input_ids']))
    print("source['offset_mapping']:", source['offset_mapping'][0][:20])
    print("source['offset_mapping'].shape:", source['offset_mapping'].shape)

    print("------------")
    print("word_relations:", word_relations)
    #input_relations = source["input_relations"].squeeze()  # .to('cuda')
    #m(source, input_relations)