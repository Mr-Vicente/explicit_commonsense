#############################
#   Imports
#############################

# Python modules
from typing import Optional, Tuple
# Remote modules
import torch
from torch import nn

# Local modules
from .attention_utils import (
    create_layer_with_commonsense_on_specific_head,
    find_head_to_mask,
    convert_relations_to_binary_mask,
    update_weights_regarding_relations_on_specific_head
)


class BartCustomAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        num_relation_kinds: int = 0,
        use_same_relation_kv_emb: bool = True,
        heads_mask:  Optional[torch.Tensor] = None,
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
        self.relation_k_emb = nn.Embedding(num_relation_kinds + 1, self.head_dim, padding_idx=0)
        if use_same_relation_kv_emb:
            self.relation_v_emb = self.relation_k_emb
        else:
            self.relation_v_emb = nn.Embedding(num_relation_kinds + 1, self.head_dim, padding_idx=0)

        self.k_rel_scale = 0.0
        self.v_rel_scale = 1.0


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

        #print('device:', hidden_states.device)
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, embed_dim = hidden_states.size()

        #print(relation_inputs.shape, 'VS ', (bsz, tgt_len, tgt_len))
        if relation_inputs is None:
            # TODO
            print('oh no: Should not come here')
            relation_inputs = torch.zeros((bsz, tgt_len, tgt_len)).to('cuda').long()
        print(relation_inputs.shape, ' | ', (bsz, tgt_len, tgt_len))
        assert relation_inputs.shape == (bsz, tgt_len, tgt_len)

        # (batch_size, seq_length, seq_length, self.num_relation_kinds, self.inner_dim // num_relation_kinds)
        relation_k_embeds = self.relation_k_emb(relation_inputs)
        relation_v_embeds = self.relation_v_emb(relation_inputs)

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
        query_states = self._shape(query_states, tgt_len, bsz)
        src_len = key_states.size(2)

        # compute scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        # q_t is [batch, seq_length, n_heads, dim_per_head]
        q_t = query_states.permute(0, 2, 1, 3)
        #print('qt.shape: ', q_t.shape)
        # r_t is [batch, seq_length, dim_per_head, seq_length]
        r_t = relation_k_embeds.transpose(-2, -1)
        #print('rt.shape: ',  r_t.shape)

        q_tr_t_matmul = torch.matmul(q_t, r_t)  # [batch, seq_length, n_heads, seq_length]
        q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)  # [batch, n_heads, seq_length, seq_length]

        # Make sure impact of relation-aware only apllicable on specific heads (k-part)

        #print("==========")
        #print('first K: ', q_tr_tmatmul_t.sum())
        """
        q_tr_tmatmul_t = self.layer_heads_relation_attention_update(
            self.heads_mask,
            q_tr_tmatmul_t,
        )
        """
        #print('second K: ', q_tr_tmatmul_t.sum())
        #print("==========")

        # give weight to influence
        #q_tr_tmatmul_t = 100.0 * q_tr_tmatmul_t

        # Add to scores
        #print('attn_weights k [before]', attn_weights)
        #print('attn_weights sum k [before]', attn_weights.sum())
        attn_weights += self.k_rel_scale * q_tr_tmatmul_t
        #attn_weights += 100.0 * q_tr_tmatmul_t
        #print('attn_weights k [after]: ', attn_weights)
        #print('attn_weights sum k [after]', attn_weights.sum())
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

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

        # Wrong place... gonna comment
        """
        attn_weights = self.layer_heads_relation_attention_update(layer_head_mask,
                                              relation_inputs,
                                              attn_weights,
                                              bsz,
                                              tgt_len,
                                              src_len)
        """
        if layer_head_mask is not None:
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

        attn_output = torch.bmm(attn_probs, value_states.view(*proj_shape))

        #print('attn_probs.shape', attn_probs.shape)
        # w_t is [batch, seq_length, n_heads, seq_length]
        w_t = attn_probs.view(bsz, self.num_heads, tgt_len, src_len).permute(0, 2, 1, 3)
        #print('w_t.shape 1:', w_t.shape)
        #print('relation_v_embeds.shape', relation_v_embeds.shape)
        # [batch, seq_length, n_heads, seq_length]
        w_tr_matmul = torch.matmul(w_t, relation_v_embeds)
        #print('w_tr_matmul.shape 1:', w_tr_matmul.shape)
        #print('w_tr_matmul.shape 2:', w_tr_matmul.shape)
        # Make sure impact of relation-aware only apllicable on specific heads (v-part)

        #print("==========")
        #print('first V sum: ', w_tr_matmul.sum())
        #print('first V: ', w_tr_matmul[0])
        """
        w_tr_matmul = self.layer_heads_relation_attention_v_update(
            self.heads_mask,
            w_tr_matmul,
            bsz,
            tgt_len,
        )
        """
        w_tr_matmul = self.v_rel_scale * w_tr_matmul
        #print('second V sum: ', w_tr_matmul.sum())
        #print('second V: ', w_tr_matmul[0])
        #print("==========")

        w_tr_matmul = w_tr_matmul.permute(0, 2, 1, 3)
        w_tr_matmul = w_tr_matmul.reshape(bsz * self.num_heads, tgt_len, self.head_dim)

        #print('attn_output v [before]', attn_output)
        #print('attn_output sum v [before]', attn_output.sum())
        attn_output += w_tr_matmul
        #attn_output += 100.0 * w_tr_matmul
        #print('attn_output v [after]', attn_output)
        #print('attn_output sum v [after]', attn_output.sum())
        #raise Exception()


        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    def layer_heads_relation_attention_update(self,
                                              layer_head_mask,
                                              data,
                                              ):
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            #print('layer_head_mask:', layer_head_mask)
            masked_weights = layer_head_mask.view(self.num_heads, 1, 1) * data
            return masked_weights
        return data

    def layer_heads_relation_attention_v_update(self,
                                              layer_head_mask,
                                              data,
                                              bsz,
                                              tgt_len,
                                              ):
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            #relation_binary_mask = convert_relations_to_binary_mask(relation_inputs)
            #one_dimension_mask = relation_binary_mask.sum(-1)
            #relation_binary_mask = convert_relations_to_binary_mask(one_dimension_mask)
            # [16, 128, 16, 64]
            masked_weights = layer_head_mask.view(self.num_heads, 1, 1) * data.view(bsz, self.num_heads, tgt_len, self.head_dim)
            return masked_weights.view(bsz, tgt_len, self.num_heads, self.head_dim)
        return data