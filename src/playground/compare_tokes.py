

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

if __name__ == '__main__':

    words_relations = [{(23, 29): {(44, 55): "related_to"}}]
    relation_kinds = ['derived_from', 'dbpedia/genre', 'antonym', 'desires', 'related_to', 'part_of',
                      'used_for', 'causes_desire', 'capable_of', 'has_context', 'entails', 'motivated_by_goal',
                      'causes', 'distinct_from', 'made_of', 'synonym', 'at_location', 'manner_of',
                      'etymologically_related_to', 'has_subevent', 'has_prerequisite', 'not_desires',
                      'has_property', 'similar_to', 'is_a', 'has_a']
    # clean_relations(read_json_file_2_dict('relation_data.json')[2])

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    question_sample = "What is happening when people seldomly have commonsense in a tube train?"
    source = tokenizer(question_sample,padding='max_length', return_tensors='pt', return_offsets_mapping=True)
    print("to normal:", tokenizer.convert_ids_to_tokens(source['input_ids'][0]))
    mappings = source['offset_mapping']
    print('mappings: ', mappings[0][:20])
    print("source['word_ids']: ", source.word_ids())
    words = source.word_ids()
    # size = (batch_idx, token_i_idx, token_j_idx)
    n_examples = len(source['input_ids'])
    aux_input_relation_kinds = np.zeros(
        (n_examples, len(source['input_ids'][0]),
         len(source['input_ids'][0])),
        dtype=np.int64
    )
    mappings = [[tuple(x) for x in mappings[idx].cpu().detach().tolist()] for idx in range(n_examples)]
    print(mappings)
    examples_mappings = []
    tokens_to_words = deque(words)
    max_idx = 0
    for mapping in mappings:
        token_idx_2_word_span = {}
        for token_idx, (char_i, char_j) in enumerate(mapping):
            word_idx_of_token = tokens_to_words.popleft()
            if word_idx_of_token is None:
                continue
            token_span = source.word_to_chars(word_idx_of_token)
            token_idx_2_word_span[token_idx] = (token_span.start, token_span.end)
            max_idx = max(token_idx, max_idx)
            examples_mappings.append(token_idx_2_word_span)
    relational_kind_to_index = {t: i + 1 for i, t in enumerate(relation_kinds)}
    for i_example in range(n_examples):
        token_idx_2_word_span = examples_mappings[i_example]
        possible_relations = words_relations[i_example]
        for token_i_idx in range(max_idx+1):
            for token_j_idx in range(max_idx + 1):
                fixed_word_range = token_idx_2_word_span.get(token_i_idx, None)
                other_word_range = token_idx_2_word_span.get(token_j_idx,None)
                if not fixed_word_range or not other_word_range:
                    continue
                relations = possible_relations.get(fixed_word_range, None)
                if not relations:
                    continue
                relation_kind = relations.get(other_word_range, None)
                if not relation_kind:
                    continue
                aux_input_relation_kinds[i_example, token_i_idx, token_j_idx] = relational_kind_to_index[relation_kind]
    # if start ==
    # aux_input_relation_kinds[batch_idx, token_i_idx, token_j_idx] =
    print('aux_input_relation_kinds:', aux_input_relation_kinds[0][5][9])
    print('aux_input_relation_kinds:', aux_input_relation_kinds[0][5][10])
    print('aux_input_relation_kinds.shape:', aux_input_relation_kinds.shape)
