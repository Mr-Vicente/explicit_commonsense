#############################
#   Imports
#############################

# Python modules
from typing import Optional, Tuple

# Remote modules
import torch
from torch import nn

# Local modules
from .attention_utils import update_weights_regarding_relations_on_specific_head


class BartCustomMaskAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        num_relation_kinds: int = 0,
        heads_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        if heads_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {heads_mask.size()}"
            )
        self.heads_mask = heads_mask

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.num_relation_kinds = num_relation_kinds


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relation_inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, embed_dim = hidden_states.size()

        #print(relation_inputs.shape, 'VS ', (bsz, tgt_len, tgt_len))
        if relation_inputs is None:
            # TODO
            relation_inputs = torch.zeros((bsz, tgt_len, tgt_len)).to('cuda').long()
        assert relation_inputs.shape == (bsz, tgt_len, tgt_len)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if self.heads_mask is not None:# and layer_head_mask is not None:
            if self.heads_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            h_mask = layer_head_mask
            #print('h_mask: ', h_mask)
            if layer_head_mask is None:
                h_mask = self.heads_mask
            #h_mask.to(attn_weights.device)
            attn_weights = update_weights_regarding_relations_on_specific_head(h_mask, attn_weights,
                                                                               relation_inputs, bsz, self.num_heads, tgt_len,
                                                                               src_len, verbose=False)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        elif layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)


        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    def find_head_to_mask(self, heads_mask) -> int:
        head_idx = torch.argmax(heads_mask)
        head_idx_simple = head_idx.item()
        return head_idx_simple

    def create_commonsense_mask(self, bsz, n_tokens, commonsense_matrix, num_heads=16, specific_head=0):
        commonsense_mask = torch.zeros(
            ((bsz, num_heads, n_tokens, n_tokens))
        )
        if commonsense_matrix is None:
            commonsense_matrix = torch.zeros(
                ((bsz, n_tokens, n_tokens))
            )
        commonsense_mask = commonsense_mask.reshape((num_heads, bsz, n_tokens, n_tokens))
        commonsense_mask[specific_head] = commonsense_matrix
        commonsense_mask = commonsense_mask.reshape((bsz, num_heads, n_tokens, n_tokens))
        return commonsense_mask

    def commonsense_attention_mask_update(self, bsz, n_tokens, commonsense_matrix, attn_weights,
                                          specific_head=0):
        num_heads = self.num_heads
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

    def convert_relations_to_binary_mask(self, input_relations):
        relations_binary_mask = input_relations.clone()
        relations_binary_mask[relations_binary_mask > 1] = 1
        return relations_binary_mask
