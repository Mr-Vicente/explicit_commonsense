from transformers import BartConfig

class BartCustomConfig(BartConfig):
    def __init__(
        self,
        model_type='bart',
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        init_std=0.02,
        classifier_dropout=0.0,
        classif_dropout=0.1,
        scale_embedding=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        forced_bos_token_id=0,
        no_repeat_ngram_size=3,  # adding
        num_hidden_layers=12,
        normalize_before=False,
        num_beams=4,
        add_bias_logits=False,
        add_final_layer_norm=False,
        early_stopping=True,
        gradient_checkpointing=False,
        num_relation_kinds = 0,
        use_same_relation_kv_emb = True,
        is_simple_mask_commonsense = False,
        should_embed_positions = False,
        heads_mask = None,
        **kwargs
    ):
        super(BartCustomConfig, self).__init__(
        model_type=model_type,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        encoder_layers=encoder_layers,
        encoder_ffn_dim=encoder_ffn_dim,
        encoder_attention_heads=encoder_attention_heads,
        decoder_layers=decoder_layers,
        decoder_ffn_dim=decoder_ffn_dim,
        decoder_attention_heads=decoder_attention_heads,
        encoder_layerdrop=encoder_layerdrop,
        decoder_layerdrop=decoder_layerdrop,
        activation_function=activation_function,
        d_model=d_model,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        init_std=init_std,
        classifier_dropout=classifier_dropout,
        classif_dropout=classif_dropout,
        scale_embedding=scale_embedding,
        use_cache=use_cache,
        num_labels=num_labels,
        pad_token_id = pad_token_id,
        bos_token_id = bos_token_id,
        eos_token_id = eos_token_id,
        is_encoder_decoder = is_encoder_decoder,
        decoder_start_token_id = decoder_start_token_id,
        forced_eos_token_id = forced_eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        no_repeat_ngram_size=no_repeat_ngram_size,  # Adding
        normalize_before=normalize_before,
        num_hidden_layers=num_hidden_layers,
        num_beams=num_beams,
        add_bias_logits=add_bias_logits,
        add_final_layer_norm=add_final_layer_norm,
        early_stopping=early_stopping,
        gradient_checkpointing=gradient_checkpointing,
        num_relation_kinds = num_relation_kinds,
        use_same_relation_kv_emb = use_same_relation_kv_emb,
        is_simple_mask_commonsense = is_simple_mask_commonsense,
        heads_mask = None,
        should_embed_positions=False,
        **kwargs
        )
        self.num_relation_kinds = num_relation_kinds
        self.use_same_relation_kv_emb = use_same_relation_kv_emb
        self.is_simple_mask_commonsense = is_simple_mask_commonsense
        self.heads_mask = heads_mask
        self.should_embed_positions = should_embed_positions

class BartSmallCustomConfig(BartConfig):
    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=6,
        encoder_ffn_dim=3072,
        encoder_attention_heads=12,
        decoder_layers=12,
        decoder_ffn_dim=3072,
        decoder_attention_heads=12,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=768,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        init_std=0.02,
        classifier_dropout=0.0,
        classif_dropout= 0.1,
        scale_embedding=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        forced_bos_token_id=0,
        no_repeat_ngram_size=3, #adding
        num_hidden_layers=6,
        normalize_before=False,
        num_beams=4,
        add_bias_logits=False,
        add_final_layer_norm=False,
        _name_or_path="bart-base",
        early_stopping=True,
        gradient_checkpointing=False,
        num_relation_kinds = 0,
        use_same_relation_kv_emb = True,
        is_simple_mask_commonsense = False,
        should_embed_positions = True,
        heads_mask = None,
        **kwargs
    ):
        super(BartSmallCustomConfig, self).__init__(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        encoder_layers=encoder_layers,
        encoder_ffn_dim=encoder_ffn_dim,
        encoder_attention_heads=encoder_attention_heads,
        decoder_layers=decoder_layers,
        decoder_ffn_dim=decoder_ffn_dim,
        decoder_attention_heads=decoder_attention_heads,
        encoder_layerdrop=encoder_layerdrop,
        decoder_layerdrop=decoder_layerdrop,
        activation_function=activation_function,
        d_model=d_model,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        init_std=init_std,
        classifier_dropout=classifier_dropout,
        classif_dropout=classif_dropout,
        scale_embedding=scale_embedding,
        use_cache=use_cache,
        num_labels=num_labels,
        pad_token_id = pad_token_id,
        bos_token_id = bos_token_id,
        eos_token_id = eos_token_id,
        is_encoder_decoder = is_encoder_decoder,
        decoder_start_token_id = decoder_start_token_id,
        forced_eos_token_id = forced_eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        no_repeat_ngram_size = no_repeat_ngram_size, #Adding
        normalize_before = normalize_before,
        num_hidden_layers=num_hidden_layers,
        num_beams=num_beams,
        add_bias_logits=add_bias_logits,
        add_final_layer_norm=add_final_layer_norm,
        _name_or_path=_name_or_path,
        early_stopping=early_stopping,
        gradient_checkpointing=gradient_checkpointing,
        num_relation_kinds = num_relation_kinds,
        use_same_relation_kv_emb = use_same_relation_kv_emb,
        is_simple_mask_commonsense = is_simple_mask_commonsense,
        heads_mask = heads_mask,
        should_embed_positions=should_embed_positions,
        **kwargs
        )
        self.num_relation_kinds = num_relation_kinds
        self.use_same_relation_kv_emb = use_same_relation_kv_emb
        self.is_simple_mask_commonsense = is_simple_mask_commonsense
        self.heads_mask = heads_mask
        self.should_embed_positions = should_embed_positions
