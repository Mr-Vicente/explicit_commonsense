
class BartConstants:
    CHECKPOINT_FOR_DOC = "facebook/bart-base"
    CONFIG_FOR_DOC = "BartConfig"
    TOKENIZER_FOR_DOC = "BartTokenizer"

    # Base model docstring
    EXPECTED_OUTPUT_SHAPE = [1, 8, 768]

    BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
        "facebook/bart-large",
    ]

    BART_START_DOCSTRING = r"""
        This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
        library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
        etc.)

        This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
        Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
        and behavior.

        Parameters:
            config ([`BartConfig`]):
                Model configuration class with all the parameters of the model. Initializing with a config file does not
                load the weights associated with the model, only the configuration. Check out the
                [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_inputs`] and
            modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information
            on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of shape
            `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids` you
            can choose to directly pass an embedded representation. This is useful if you want more control over how to
            convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
    BART_GENERATION_EXAMPLE = r"""
        Summarization example:

        ```python
        >>> from transformers import BartTokenizer, BartForConditionalGeneration

        >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

        >>> ARTICLE_TO_SUMMARIZE = (
        ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
        ... )
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
        >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
        ```

        Mask filling example:

        ```python
        >>> from transformers import BartTokenizer, BartForConditionalGeneration

        >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

        >>> TXT = "My friends are <mask> but they eat too many carbs."
        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        ['not', 'good', 'healthy', 'great', 'very']
        ```
    """



