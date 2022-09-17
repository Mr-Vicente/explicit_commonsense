# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import List, Optional, Tuple, Dict
from collections import deque

import torch
import numpy as np

from tokenizers import pre_tokenizers, processors

from transformers.tokenization_utils_base import AddedToken, BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging
from transformers.models.bart.tokenization_bart import BartTokenizer


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# See all BART models at https://huggingface.co/models?filter=bart
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/vocab.json",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/vocab.json",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/vocab.json",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/vocab.json",
    },
    "merges_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/merges.txt",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/merges.txt",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/merges.txt",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/merges.txt",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/tokenizer.json",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/tokenizer.json",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/tokenizer.json",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/tokenizer.json",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/bart-base": 1024,
    "facebook/bart-large": 1024,
    "facebook/bart-large-mnli": 1024,
    "facebook/bart-large-cnn": 1024,
    "facebook/bart-large-xsum": 1024,
    "yjernite/bart_eli5": 1024,
}


class BartCustomTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" BART tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2 tokenizer,
    using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import BartTokenizerFast
    >>> tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    >>> tokenizer("Hello world")['input_ids']
    [0, 31414, 232, 2]
    >>> tokenizer(" Hello world")['input_ids']
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (BART tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask", "input_commonsense_relations", "commonsense_mask"]
    slow_tokenizer_class = BartTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )

        self.relational_kind_to_index = None
        self.there_is_difference_between_relations = True

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

        # the pre_tokenizer is already updated in the GPT2TokenizerFast `__init__`
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # The lists 'sep' and 'cls' must be cased in tuples for the object `post_processor_class`
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

    def __call__(self, *args, **kwargs):
        input_commonsense_relations = kwargs.get('input_commonsense_relations', None)
        if 'input_commonsense_relations' in kwargs:
            kwargs.pop('input_commonsense_relations')
        out = super(BartCustomTokenizerFast, self).__call__(*args, **kwargs)
        if out.get('input_commonsense_relations') is None:
            out = self._post_process_tokenization(input_commonsense_relations, out)
        return out

    def set_known_relation_names(self, known_relations_names: List[str]):
        self.relational_kind_to_index = {t: i + 1 for i, t in enumerate(known_relations_names)}

    def set_operation_mode(self, there_is_difference_between_relations=True):
        self.there_is_difference_between_relations = there_is_difference_between_relations

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        BART tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *<mask>*.
        """
        if self._mask_token is None and self.verbose:
            logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on Bart.
        """
        # Mask token behave like a normal word, i.e. include the space before it
        # So we set lstrip to True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )
        input_commonsense_relations = kwargs.get('input_commonsense_relations', None)
        if 'input_commonsense_relations' in kwargs:
            kwargs.pop('input_commonsense_relations')
        out = super()._batch_encode_plus(*args, **kwargs)
        if out.get('input_commonsense_relations') is None:
            out = self._post_process_tokenization(input_commonsense_relations, out)
        return out

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        input_commonsense_relations = kwargs.get('input_commonsense_relations', None)
        if 'input_commonsense_relations' in kwargs:
            kwargs.pop('input_commonsense_relations')
        out = super()._encode_plus(*args, **kwargs)
        if out.get('input_commonsense_relations') is None:
            out = self._post_process_tokenization(input_commonsense_relations, out)
        return out

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BART does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _post_process_tokenization(self, input_commonsense_relations, out: BatchEncoding) -> BatchEncoding:
        new_input_relations = self.get_new_input_relation_kinds(
            tokenizer_outputs=out, input_relations=input_commonsense_relations
        )
        #if new_input_relations is not None:
        #    print('sum:', new_input_relations.sum())
        out['input_commonsense_relations'] = new_input_relations
        return out

    def find_new_tokens_span_for_multiword(self, pair, aux_dict):
        old_start, old_end = pair
        #print('pair:', pair)
        keys = list(aux_dict.keys())
        #print('aux_dict:', aux_dict)
        new_start, new_end = old_start, old_end
        for (start, end) in keys:
            #print('-----> (start, end)', (start, end))
            #print('old_start, old_end:', old_start, old_end)
            #print('start, end:', start, end)
            if old_start >= start and old_end <= end:
                new_start, new_end = start, end
                break
        return new_start, new_end

    def find_new_tokens_incoming_span_for_multiword(self, pair, aux_dict):
        old_start, old_end = pair
        incoming_rels = list([coord for v in aux_dict.values() for coord, relation in v.items()])
        new_start, new_end = old_start, old_end
        for (start, end) in incoming_rels:
            #print('-----> (start, end)', (start, end))
            #print('old_start, old_end:', old_start, old_end)
            #print('start, end:', start, end)
            if old_start >= start and old_end <= end:
                new_start, new_end = start, end
                break
        return new_start, new_end

    def get_new_input_relation_kinds(
            self,
            tokenizer_outputs: BatchEncoding,
            input_relations: Optional[List[Dict[Tuple[int, int], Dict[Tuple[int, int], str]]]] = None
    ) -> torch.Tensor:

        n_examples = len(tokenizer_outputs['input_ids'])
        n_tokens = len(tokenizer_outputs['input_ids'][0])
        aux_input_relation_kinds = np.zeros(
            (n_examples, n_tokens, n_tokens),
            dtype=np.int64
        )
        if not input_relations and input_relations is not None:
            return torch.from_numpy(aux_input_relation_kinds)
        elif not input_relations:
            return None#torch.tensor([])
        assert 'offset_mapping' in tokenizer_outputs, "Run tokenizer with return_offsets_mapping=True"
        # print('aux_input_relation_kinds.shape', tokenizer_outputs['input_ids'].shape)
        #print('input_relations:', input_relations)
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
                #print(idx, mapping)
                words = tokenizer_outputs.word_ids(batch_index=idx)
                tokens_to_words = deque(words)
                token_idx_2_word_span = {}
                for token_idx, (_char_i, _char_j) in enumerate(mapping):
                    word_idx_of_token = tokens_to_words.popleft()
                    if word_idx_of_token is None:
                        continue
                    token_span = tokenizer_outputs.word_to_chars(word_idx_of_token)
                    token_idx_2_word_span[token_idx] = (token_span.start, token_span.end) # sera que tenho de tirar o menos 1 (estava -1)
                    max_idx = max(token_idx, max_idx)
                #print('token_idx_2_word_span:', token_idx_2_word_span)
                ##### Multiword ######
                token_idx_2_word_span_multiword = {}
                d = input_relations[idx]
                for k, v in token_idx_2_word_span.items():
                    #print('k,v', k, v)
                    new_start, new_end = self.find_new_tokens_span_for_multiword(v, d)
                    token_idx_2_word_span_multiword[k] = (new_start, new_end)
                    #print('tmp:', token_idx_2_word_span_multiword)
                    #print('[before]token_idx_2_word_span_multiword[k]:', token_idx_2_word_span_multiword[k])
                    if v[0]==new_start and v[1]==new_end:
                        new_start, new_end = self.find_new_tokens_incoming_span_for_multiword(v, d)
                        token_idx_2_word_span_multiword[k] = (new_start, new_end)
                    #print('tmp2:', token_idx_2_word_span_multiword)
                    #print('[after]token_idx_2_word_span_multiword[k]:', token_idx_2_word_span_multiword[k])
                #####           ######
                #print('token_idx_2_word_span_multiword:', token_idx_2_word_span_multiword)
                examples_mappings.append(token_idx_2_word_span_multiword)
            # print('len:', len(examples_mappings))
            # print('max_idx: ', max_idx)
            for i_example in range(n_examples):
                token_idx_2_word_span = examples_mappings[i_example]
                # print('token_idx_2_word_span: ', token_idx_2_word_span)
                possible_relations = input_relations[i_example]
                # print('possible_relations: ', possible_relations)
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
                        #print('relation_kind:',relation_kind)
                        if self.there_is_difference_between_relations:
                            aux_input_relation_kinds[i_example, token_i_idx, token_j_idx] = self.relational_kind_to_index[relation_kind]
                        else:
                            # basic relation | only matters that relation exists between tokens
                            aux_input_relation_kinds[i_example, token_i_idx, token_j_idx] = 1
        aux_input_relation_kinds = torch.from_numpy(aux_input_relation_kinds)
        return aux_input_relation_kinds

    def create_commonsense_mask(self, tokenizer_outputs, commonsense_matrix, num_heads=16, specific_head=0):
        bsz = len(tokenizer_outputs['input_ids'])
        n_tokens = len(tokenizer_outputs['input_ids'][0])
        commonsense_mask = np.zeros(
            ((bsz, num_heads, n_tokens, n_tokens)),
            dtype=np.int64
        )
        if commonsense_matrix is None:
            commonsense_matrix = np.zeros(
                ((bsz, n_tokens, n_tokens)),
                dtype=np.int64
            )
        commonsense_mask = commonsense_mask.reshape((num_heads, bsz, n_tokens, n_tokens))
        # commonsense_matrix.shape: (bsz, src_len, tgt_len)
        #print('commonsense_matrix:', commonsense_matrix)
        commonsense_mask[specific_head] = commonsense_matrix
        commonsense_mask = commonsense_mask.reshape((bsz, num_heads, n_tokens, n_tokens))
        return commonsense_mask
