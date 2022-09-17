#############################
#   Imports
#############################

# Python modules

# Remote modules
import torch

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

def find_head_to_mask(heads_mask) -> int:
    head_idx = torch.argmax(heads_mask)
    head_idx_simple = head_idx.item()
    return head_idx_simple

def commonsense_attention_mask_update(bsz, n_tokens, commonsense_matrix, attn_weights,
                                      num_heads=16, specific_head=0):
    commonsense_mask = torch.zeros(
        ((bsz, num_heads, n_tokens, n_tokens))
    )
    attn_weights_helper = attn_weights.reshape((num_heads, bsz, n_tokens, n_tokens))
    zeros = torch.zeros(
        ((bsz, n_tokens, n_tokens))
    )
    head_previous_attention_weights = attn_weights_helper[specific_head]
    attn_weights_helper[specific_head] = zeros
    attn_weights_helper = attn_weights_helper.reshape((bsz, num_heads, n_tokens, n_tokens))
    if commonsense_matrix is None:
        # ignore is not passed (ones -> neutral since multiplication is used)
        commonsense_matrix = torch.ones(
            ((bsz, n_tokens, n_tokens))
        )
    commonsense_mask = commonsense_mask.reshape((num_heads, bsz, n_tokens, n_tokens))
    commonsense_mask[specific_head] = head_previous_attention_weights * commonsense_matrix
    # TODO Stupid conversion
    commonsense_mask = commonsense_mask.reshape((bsz, num_heads, n_tokens, n_tokens)).to('cuda')
    return attn_weights_helper + commonsense_mask

def convert_relations_to_binary_mask(input_relations, should_clone=True):
    relations_binary_mask=input_relations
    if should_clone:
        relations_binary_mask = input_relations.clone()
    relations_binary_mask[relations_binary_mask > 1] = 1
    return relations_binary_mask

def relation_binary_2d_to_1d(relations_binary_mask):
    relations_binary_mask = relations_binary_mask.sum(dim=1)
    relations_binary_mask[relations_binary_mask > 1] = 1
    return relations_binary_mask

def create_layer_with_commonsense_on_specific_head(relation_binary_mask, bsz, num_heads, specific_head=0):
    n_tokens = relation_binary_mask.size()[-1]
    relations_mask = torch.zeros(
        (bsz, num_heads, n_tokens, n_tokens)
    )
    layer = relations_mask.reshape((num_heads, bsz, n_tokens, n_tokens))
    layer[specific_head] = relation_binary_mask
    layer = layer.reshape((bsz, num_heads, n_tokens, n_tokens))
    return layer

def update_weights_regarding_relations_on_specific_head(layer_head_mask, attn_weights, relation_inputs, bsz, num_heads, tgt_len, src_len, verbose=True):
    #layer_head_mask = layer_head_mask.to(attn_weights.device)
    inverse_layer_head_mask = (layer_head_mask.view(num_heads, 1, 1) - 1) * -1
    #inverse_layer_head_mask = inverse_layer_head_mask.to(attn_weights.device)
    #print('layer_head_mask:', layer_head_mask)
    if verbose:
        print("==============================")
        print('layer_head_mask.shape:',  layer_head_mask.shape)
        print('inverse_layer_head_mask.shape:',  inverse_layer_head_mask.shape)
        print('attn_weights.shape:',  attn_weights.shape)
        print('relation_inputs.shape', relation_inputs.shape)
        print("==============================")
    #print('layer_head_mask.device:', layer_head_mask.device)
    #print('inverse_layer_head_mask.device:', inverse_layer_head_mask.device)
    #print('relation_inputs.device:', relation_inputs.device)
    intermediate_weights = inverse_layer_head_mask * attn_weights.view(bsz, num_heads, tgt_len, src_len)
    relation_inputs = convert_relations_to_binary_mask(relation_inputs, should_clone=False)
    relation_weights = layer_head_mask.view(num_heads, 1, 1) * relation_inputs.view(bsz,1,tgt_len, src_len) * attn_weights.view(bsz, num_heads,
                                                                                               tgt_len, src_len)
    attn_weights = intermediate_weights + relation_weights
    # [batch, n_heads, seq_length, seq_length]
    if verbose:
        print('attn_weights_int.shape', attn_weights.shape)
    return attn_weights