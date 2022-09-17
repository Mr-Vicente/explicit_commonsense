# Python modules

# Remote modules
import numpy as np
from transformers import (
    BartForConditionalGeneration,
)


# Local modules
from heads_analysis.attention_viz import AttentionVisualizer
from data.RelationsDatasets import RelationsDataset
from data.relation_utils import clean_relations
from data.data_preparation import load_and_preprocess_data
from utils import Data_Type, KGType, get_device

from kgs_binding.kg_qa_binding_utils import get_kg_qa_data_metadata, from_relations_path_2_relations, load_kg_handler

from custom_tokenizer.bart_custom_tokenizer_fast import BartCustomTokenizerFast

#############################
#   Constants
#############################

model_name = 'facebook/bart-large'

def batch_index_to_str(batch, relations_info):
    new_batch = []
    for i, b in enumerate(batch):
        new_batch.append(index_to_str(b, relations_info[i]))
    return new_batch


def index_to_str(text, relations_info):
    words = []
    for (i,j), v in relations_info.items():
        words.append(text[i,j])
        for (i_other, j_other), rel in v.items():
            words.append(text[i_other, j_other])
    return list(set(words))


def load_tokenizer_and_model(knowledge_handler, device):
    #tokenizer = BartTokenizer.from_pretrained(model_name)
    tokenizer = BartCustomTokenizerFast.from_pretrained(model_name)
    relation_names = knowledge_handler.get_relation_types()
    tokenizer.set_known_relation_names(relation_names)
    tokenizer.set_operation_mode(there_is_difference_between_relations=True)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    model.zero_grad()
    model.to(device)
    return tokenizer, model

def predict(inputs, attention_mask=None):
    output = model(inputs, attention_mask=attention_mask, output_attentions=True)
    loss = output.loss
    logits = output.logits
    past_key_values = output.past_key_values
    decoder_hidden_states = output.decoder_hidden_states
    decoder_attentions = output.decoder_attentions
    cross_attentions = output.cross_attentions
    encoder_last_hidden_state = output.encoder_last_hidden_state
    encoder_hidden_states = output.encoder_hidden_states
    encoder_attentions = output.encoder_attentions
    return loss, logits, encoder_attentions

if __name__ == '__main__':
    att_viz = AttentionVisualizer()
    device = get_device()

    dataset_types = [Data_Type.ELI5, Data_Type.STACK_EXCHANGE]
    knowledge_handler = load_kg_handler(kg_type=KGType.CONCEPTNET)
    tokenizer, model = load_tokenizer_and_model(knowledge_handler, device)

    training_data, validation_data, test_data = load_and_preprocess_data(dataset_types, limit=500)
    relations_metadata = get_kg_qa_data_metadata(knowledge_handler)
    relations_data = from_relations_path_2_relations(dataset_types, metadata=relations_metadata)
    dataset = RelationsDataset(training_data, relations_data=relations_data, tokenizer=tokenizer, device=device)

    start = 18
    batch = dataset.tokenized_data[start:start + 2]
    batch_relations_data = dataset.word_relations
    normal_data_batch = dataset.data[start:start + 2]
    batch_relations_data = [batch_relations_data[x[0]] for x in normal_data_batch]
    normal_batch = [d[-1]['question'] for d in dataset.data[start:start + 2]]
    print(batch_index_to_str(normal_batch, batch_relations_data))
    print(batch)
    for inputs in batch:
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        input_commonsense_relations = inputs.get("input_commonsense_relations")
        if 'input_commonsense_relations' in inputs:
            input_commonsense_relations = input_commonsense_relations.clone()
            input_commonsense_relations[input_commonsense_relations > 1] = 1
            inputs.pop('input_commonsense_relations')
        loss, logits, encoder_attentions = predict(inputs)
    """
    batch_input_ids = [unit['input_ids'].to(device) for unit in batch]
    loss, logits, encoder_attentions = predict(batch_input_ids)
    print('encoder_attentions.shape:', encoder_attentions.shape)
    print('encoder_attentions:', encoder_attentions)
    """

