
import torch
from context_2_matrix import context_2_matrix

def commonsense_attention_mask_update(bsz, n_tokens, commonsense_matrix, attn_weights, num_heads=16, specific_head=0):
    commonsense_mask = torch.zeros(
        ((bsz, num_heads, n_tokens, n_tokens))
    )
    attn_weights_helper = attn_weights.reshape((num_heads, bsz, n_tokens, n_tokens))
    zeros = torch.zeros(
        ((bsz, n_tokens, n_tokens))
    )
    head_previous_attention_wights = attn_weights_helper[specific_head]
    attn_weights_helper[specific_head] = zeros
    attn_weights_helper = attn_weights_helper.reshape((bsz, num_heads, n_tokens, n_tokens))
    if commonsense_matrix is None:
        # ignore is not passed (ones -> neutral since multiplication is used)
        commonsense_matrix = torch.ones(
            ((bsz, n_tokens, n_tokens))
        )
    commonsense_mask = commonsense_mask.reshape((num_heads, bsz, n_tokens, n_tokens))
    commonsense_mask[specific_head] = head_previous_attention_wights * commonsense_matrix
    commonsense_mask = commonsense_mask.reshape((bsz, num_heads, n_tokens, n_tokens))
    return attn_weights_helper + commonsense_mask

def commonsense_attention_mask_update(bsz, n_tokens, commonsense_matrix, attn_weights,
                                      specific_head=0):
    num_heads = 16
    commonsense_mask = torch.zeros(
        ((bsz, num_heads, n_tokens, n_tokens))
    )
    attn_weights_helper = attn_weights.reshape((num_heads, bsz, n_tokens, n_tokens))
    zeros = torch.zeros(
        ((bsz, n_tokens, n_tokens))
    )
    #head_previous_attention_weights = attn_weights_helper[specific_head]
    #attn_weights_helper[specific_head] = zeros
    attn_weights_helper = attn_weights_helper.reshape((bsz, num_heads, n_tokens, n_tokens))
    if commonsense_matrix is None:
        # ignore is not passed (ones -> neutral since multiplication is used)
        commonsense_matrix = torch.ones(
            ((bsz, n_tokens, n_tokens))
        )
    commonsense_mask = commonsense_mask.reshape((num_heads, bsz, n_tokens, n_tokens))
    commonsense_mask[specific_head] = head_previous_attention_weights * commonsense_matrix
    # TODO Stupid conversion
    commonsense_mask = commonsense_mask.reshape((bsz, num_heads, n_tokens, n_tokens))
    return attn_weights_helper + commonsense_mask

if __name__ == '__main__':
    bsz = 2
    n_tokens = 128
    attn_weights = torch.randn((bsz, 16, n_tokens, n_tokens))
    commonsense_matrix = context_2_matrix()
    commonsense_matrix = torch.from_numpy(commonsense_matrix)
    new_mask = commonsense_attention_mask_update(bsz, n_tokens, commonsense_matrix, attn_weights)
    print('new_mask: ', new_mask)
    x = 10
