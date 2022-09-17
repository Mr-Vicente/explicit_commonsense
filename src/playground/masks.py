
import torch as t
import numpy as np

from context_2_matrix import context_2_matrix

def create_commonsense_mask(commonsense_matrix, num_heads=16, specific_head=0):
    commonsense_mask = np.zeros(
        ((bsz, num_heads, tgt_len, src_len)),
        dtype=np.int64
    )
    commonsense_mask =  commonsense_mask.reshape((num_heads, bsz, src_len, tgt_len))
    # commonsense_matrix.shape: (bsz, src_len, tgt_len)
    commonsense_mask[specific_head] = commonsense_matrix
    commonsense_mask =  commonsense_mask.reshape((bsz, num_heads, src_len, tgt_len))
    return commonsense_mask

if __name__ == '__main__':
    bsz = 2
    tgt_len = 128
    src_len = 128
    commonsense_matrix = context_2_matrix()
    commonsense_mask = create_commonsense_mask(commonsense_matrix)
    mask = np.ones(
        ((bsz, 16, tgt_len, src_len)),
        dtype=np.int64
    )
    torch_commonsense_mask = t.from_numpy(commonsense_mask)
    torch_mask = t.from_numpy(mask)
    new_mask = torch_commonsense_mask * torch_mask
    print('new_mask: ', new_mask)