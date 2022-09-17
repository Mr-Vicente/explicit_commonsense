
import numpy as np
import torch
from transformers import (
    BartTokenizerFast, BatchEncoding,
)
import matplotlib.pyplot as plt

from collections import deque
from ast import literal_eval

def change_dict(pair, aux_dict):
    old_start, old_end = pair
    keys = list(aux_dict.keys())
    new_start, new_end = old_start, old_end
    for (start, end) in keys:
        #print('old_start, old_end:', old_start, old_end)
        #print('start, end:', start, end)
        if old_start >= start and old_end <= end:
            new_start, new_end = start, end
            break
    return new_start, new_end



def draw_map(all_tokens, attns, relations_types):
    all_tokens = [tok.replace('Ġ', '') for tok in all_tokens]
    font_size = 18
    plt.clf()
    plt.close('all')
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    fig, ax = plt.subplots(figsize=(18,15))
    plt.title('Concepts Relationship', fontsize=font_size)
    cmap = plt.get_cmap('PiYG', len(relations_types))
    plt.imshow(attns, interpolation='none', cmap=cmap,vmin=0 - 0.5,
                      vmax=len(relations_types)-1 + 0.5)
    plt.xticks(range(len(all_tokens)), all_tokens, rotation=90)
    plt.yticks(range(len(all_tokens)), all_tokens)
    def rect(pos):
        r = plt.Rectangle(pos - 0.5, 1, 1, facecolor="none", edgecolor="k", linewidth=2)
        plt.gca().add_patch(r)

    x, y = np.meshgrid(np.arange(attns.shape[1]), np.arange(attns.shape[0]))
    m = np.c_[x[attns==0], y[attns==0]]
    for pos in m:
        rect(pos)
    cbar = plt.colorbar(label='attention weights')
    cbar.set_ticks(range(len(relations_types)))
    cbar.set_ticklabels(relations_types, fontsize=font_size)
    #plt.show()
    plt.savefig(f'figs/concepts_relations.png', dpi=fig.dpi)

def visualize_token2token_scores(all_tokens,
                                 scores_mat,
                                 useful_indeces,
                                 x_label_name='Head',
                                 apply_normalization=True,
                                 relations_types=None):
    #fig = plt.figure(figsize=(20, 20))
    all_tokens = np.array(all_tokens)[useful_indeces].tolist()
    if apply_normalization:
        scores = torch.from_numpy(scores_mat)
        shape = scores.shape
        scores = scores.reshape((shape[0], shape[1], 1))
        scores_mat = torch.linalg.norm(scores, dim=2)
    scores_np = np.array(scores_mat)[0]
    scores_np = scores_np[useful_indeces, :]
    scores_np = scores_np[:, useful_indeces]
    draw_map(all_tokens, scores_np, relations_types)

def get_new_input_relation_kinds(
        tokenizer_outputs: BatchEncoding,
        input_relations = None,
        known_relations_names = []
) -> torch.Tensor:

    assert 'offset_mapping' in tokenizer_outputs, "Run tokenizer with return_offsets_mapping=True"
    n_examples = len(tokenizer_outputs['input_ids'])
    n_tokens = len(tokenizer_outputs['input_ids'][0])
    relational_kind_to_index = {t: i + 1 for i, t in enumerate(known_relations_names)}
    aux_input_relation_kinds = np.zeros(
        (n_examples, n_tokens, n_tokens),
        dtype=np.int64
    )
    if not input_relations and input_relations is not None:
        return torch.from_numpy(aux_input_relation_kinds)
    elif not input_relations:
        return None
    # print('aux_input_relation_kinds.shape', tokenizer_outputs['input_ids'].shape)
    print('input_relations:', input_relations)
    if input_relations is not None:
        # if input_relations is dirty, clean it
        if isinstance(input_relations, dict):
            input_relations = [input_relations]
        mappings = tokenizer_outputs['offset_mapping']
        assert len(mappings) == len(input_relations)
        # print("to normal:", self.tokenizer.convert_ids_to_tokens(tokenizer_outputs['input_ids'][0]))
        # print('words: ', words)
        # print('x: ', mappings)
        mappings = [[tuple(x) for x in mappings[idx].cpu().detach().tolist()] for idx in range(n_examples)]
        # print(mappings)
        examples_mappings = []
        max_idx = 0
        for idx, mapping in enumerate(mappings):
            print(idx, mapping)
            words = tokenizer_outputs.word_ids(batch_index=idx)
            print('words:', words)
            tokens_to_words = deque(words)
            token_idx_2_word_span = {}
            for token_idx, (_char_i, _char_j) in enumerate(mapping):
                word_idx_of_token = tokens_to_words.popleft()
                if word_idx_of_token is None:
                    continue
                token_span = tokenizer_outputs.word_to_chars(word_idx_of_token)
                token_idx_2_word_span[token_idx] = (token_span.start, token_span.end)
                max_idx = max(token_idx, max_idx)
            ##### Multiword ######
            token_idx_2_word_span_multiword = {}
            d = input_relations[idx]
            for k, v in token_idx_2_word_span.items():
                new_start, new_end = change_dict(v, d)
                token_idx_2_word_span_multiword[k] = (new_start, new_end)
            #####           ######
            examples_mappings.append(token_idx_2_word_span_multiword)
        print('len:', len(examples_mappings))
        print('max_idx: ', max_idx)
        print('examples_mappings: ', examples_mappings)
        for i_example in range(n_examples):
            token_idx_2_word_span = examples_mappings[i_example]
            print('token_idx_2_word_span: ', token_idx_2_word_span)
            possible_relations = input_relations[i_example]
            print('possible_relations: ', possible_relations)
            for token_i_idx in range(max_idx + 1):
                for token_j_idx in range(max_idx + 1):
                    fixed_word_range = token_idx_2_word_span.get(token_i_idx, None)
                    other_word_range = token_idx_2_word_span.get(token_j_idx, None)
                    if not fixed_word_range or not other_word_range:
                        continue
                    #print(fixed_word_range, ' | ', other_word_range)
                    relations = possible_relations.get(fixed_word_range, None)
                    if not relations:
                        continue
                    #print('possible_relations:' , possible_relations)
                    relation_kind = relations.get(other_word_range, None)
                    if not relation_kind:
                        continue
                    aux_input_relation_kinds[i_example, token_i_idx, token_j_idx] = \
                        relational_kind_to_index[
                            relation_kind]
    aux_input_relation_kinds = torch.from_numpy(aux_input_relation_kinds).to('cpu')
    return aux_input_relation_kinds

if __name__ == '__main__':
    #race car
    input_relations = [{(14, 22): {(42, 46): 'related_to'}, (42, 46): {(14, 22): 'related_to'}}]
    # plants
    #input_relations = [{(16, 21): {(26, 32): 'related_to'}, (26, 32): {(16, 21): 'related_to', (41, 46): 'related_to'}, '(41, 46)': {'(26, 32)': 'related_to'}}]
    relation_kinds = ['no_relation', 'derived_from', 'dbpedia/genre', 'antonym', 'desires', 'related_to', 'part_of',
                      'used_for', 'causes_desire', 'capable_of', 'has_context', 'entails', 'motivated_by_goal',
                      'causes', 'distinct_from', 'made_of', 'synonym', 'at_location', 'manner_of',
                      'etymologically_related_to', 'has_subevent', 'has_prerequisite', 'not_desires',
                      'has_property', 'similar_to', 'is_a', 'has_a']
    # clean_relations(read_json_file_2_dict('relation_data.json')[2])

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    question_sample = 'i want have a race car like the ones very fast of nascar!'
    #question_sample = 'i would like to water the plants from my house'
    tokenizer_outputs = tokenizer(question_sample, padding='max_length',
                                  return_tensors='pt', return_offsets_mapping=True, max_length=128)
    matrix = get_new_input_relation_kinds(tokenizer_outputs, input_relations, relation_kinds)
    print('matrix: ', matrix)
    print('sum:', matrix.sum())
    input_ids = tokenizer_outputs['input_ids']
    indices = input_ids[0].detach().numpy()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    useful_indeces = indices != tokenizer.pad_token_id
    visualize_token2token_scores(all_tokens=all_tokens,
                                 scores_mat=matrix,
                                 useful_indeces=useful_indeces,
                                 apply_normalization=False,
                                 relations_types=relation_kinds,
    )
    """
    d = {(16, 21): {(26, 32): 'related_to'}, (26, 32): {(16, 21): 'related_to', (41, 46): 'related_to'}, (41, 46): {(26, 32): 'related_to'}}
    d = {(14, 22): {(42, 46): 'related_to'}, (42, 46): {(14, 22): 'related_to'}}
    a = {1: (0, 1), 2: (2, 7), 3: (8, 12), 4: (13, 15), 5: (16, 21), 6: (22, 25), 7: (26, 32), 8: (33, 37), 9: (38, 40), 10: (41, 46)}
    a = {1: (0, 1), 2: (2, 6), 3: (7, 11), 4: (12, 13), 5: (14, 18), 6: (19, 22), 7: (23, 27), 8: (28, 31), 9: (32, 36), 10: (37, 41), 11: (42, 46), 12: (47, 49), 13: (50, 56), 14: (50, 56), 15: (56, 57)}
    b = {}
    for k,v in a.items():
        new_start, new_end = change_dict(v, d)
        b[k] = (new_start, new_end)
    print('old:', a)
    print('new:', b)
    """