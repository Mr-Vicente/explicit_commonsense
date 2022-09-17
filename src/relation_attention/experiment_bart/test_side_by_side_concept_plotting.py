# Python modules

# Remote modules
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import (
    BartForConditionalGeneration,
    default_data_collator
)
from transformers.tokenization_utils_base import BatchEncoding
from torch.utils.data import DataLoader


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

def plot_attn_lines_concepts(title, examples, layer, head, color_words,
              color_from=True, width=3, example_sep=3,
              word_height=1, pad=0.1, hide_sep=False):
    # examples -> {'words': tokens, 'attentions': [layer][head]}
    plt.clf()
    plt.figure(figsize=(6, 5))
    for i, example in enumerate(examples):
        yoffset = 0
        if i == 0:
            yoffset += (len(examples[0]["words"]) -
                        len(examples[1]["words"])) * word_height / 2
        xoffset = i * width * example_sep
        attn = example["attentions"][layer][head]
        if hide_sep:
            attn = np.array(attn)
            attn[:, 0] = 0
            attn[:, -1] = 0
            attn /= attn.sum(axis=-1, keepdims=True)

        words = example["words"]
        n_words = len(words)
        for position, word in enumerate(words):
            for x, from_word in [(xoffset, True), (xoffset + width, False)]:
                color = "k"
                if from_word == color_from and word in color_words:
                    color = "#cc0000"
                plt.text(x, yoffset - (position * word_height), word,
                         ha="right" if from_word else "left", va="center",
                         color=color)

        for i in range(n_words):
            for j in range(n_words):
                color = "b"
                if words[i if color_from else j] in color_words:
                    color = "r"
                plt.plot([xoffset + pad, xoffset + width - pad],
                         [yoffset - word_height * i, yoffset - word_height * j],
                         color=color, linewidth=1, alpha=attn[i, j])
    plt.axis("off")
    plt.title(title)
    plt.show()

def plot_attn_lines_concepts_ids(title, examples, layer, head, color_words,
              relations_total,
              color_from=True, width=3, example_sep=3,
              word_height=1, pad=0.1, hide_sep=False):
    # examples -> {'words': tokens, 'attentions': [layer][head]}
    plt.clf()
    plt.figure(figsize=(6, 5))
    print('relations_total:', relations_total)
    for idx, example in enumerate(examples):
        yoffset = 0
        if idx == 0:
            yoffset += (len(examples[0]["words"]) -
                        len(examples[1]["words"])) * word_height / 2
        xoffset = idx * width * example_sep
        attn = example["attentions"][layer][head]
        if hide_sep:
            attn = np.array(attn)
            attn[:, 0] = 0
            attn[:, -1] = 0
            attn /= attn.sum(axis=-1, keepdims=True)

        words = example["words"]
        n_words = len(words)
        example_rel = relations_total[idx]
        print('color_words:', color_words)
        for position, word in enumerate(words):
            for x, from_word in [(xoffset, True), (xoffset + width, False)]:
                color = "k"
                for y_idx, y in enumerate(words):
                    if from_word and example_rel[position, y_idx] > 0 and position <= y_idx:
                        color = "r"
                    if not from_word and example_rel[position, y_idx] > 0 and position >= y_idx:
                        color = "g"
                #if from_word == color_from and word in color_words:
                #    color = "#cc0000"
                plt.text(x, yoffset - (position * word_height), word,
                         ha="right" if from_word else "left", va="center",
                         color=color)

        for i in range(n_words):
            for j in range(n_words):
                color = "b"
                #print(i,j, example_rel[i,j])
                if example_rel[i,j].item() > 0 and i <= j:
                    color = "r"
                if example_rel[i,j].item() > 0 and i >= j:
                    color = "g"
                plt.plot([xoffset + pad, xoffset + width - pad],
                         [yoffset - word_height * i, yoffset - word_height * j],
                         color=color, linewidth=1, alpha=attn[i, j])
    plt.axis("off")
    plt.title(title)
    plt.show()

def relations_info_to_concepts_str(text, relations_info):
    words = []
    print(text)
    for (i,j), v in relations_info.items():
        #print(i,j)
        words.append(text[i:j])
        for (i_other, j_other), rel in v.items():
            words.append(text[i_other: j_other])
            print(f'{text[i:j]} -> {text[i_other: j_other]}')
    return list(set(words))

def batch_index_to_str(batch, relations_info):
    new_batch = []
    for i, b in enumerate(batch):
        new_batch.append(relations_info_to_concepts_str(b, relations_info[i]))
    return new_batch

def predict(model, inputs):
    output = model(**inputs, output_attentions=True)
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

def get_examples(model, device, tokenizer, dataloader):
    examples_total = []
    relations_total = []
    for step, inputs in enumerate(dataloader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        input_commonsense_relations = inputs.get("input_commonsense_relations")
        if 'input_commonsense_relations' in inputs:
            inputs.pop('input_commonsense_relations')
        loss, logits, encoder_attentions = predict(model, inputs)
        encoder_attentions = torch.stack(encoder_attentions)
        print(encoder_attentions.size())
        n_layers, batch_size, n_heads, src, tgt = encoder_attentions.size()
        encoder_attentions = encoder_attentions.view(batch_size, n_layers, n_heads, src, tgt)
        examples = []
        for i, ex in enumerate(encoder_attentions):
            d = {}
            indices = inputs['input_ids'][i].detach().cpu()
            all_tokens = tokenizer.convert_ids_to_tokens(indices)
            useful_indeces = indices != tokenizer.pad_token_id
            all_tokens = np.array(all_tokens)[useful_indeces]
            all_tokens = [tok.replace('Ġ', '') for tok in all_tokens]
            d['words'] = all_tokens
            d['attentions'] = ex.detach().cpu().numpy()
            examples.append(d)
        print(d['words'])
        relations_total.append(input_commonsense_relations)
        examples_total.append(examples)
    return examples_total, relations_total


def input_configation(dataset):
    start = 18
    batch = dataset.tokenized_data[start:start + 2]
    batch_relations_data = dataset.word_relations
    # print('batch_relations_data:', batch_relations_data[:2])
    # print(batch)
    normal_data_batch = dataset.data[start:start + 2]
    batch_relations_data = [batch_relations_data[x[0]] for x in normal_data_batch]
    # normal_batch = [tokenizer.decode(b['input_ids'], skip_special_tokens=True) for b in batch]
    normal_batch = [d[-1]['question'] for d in dataset.data[start:start + 2]]
    words_to_spot = [word for words in batch_index_to_str(normal_batch, batch_relations_data) for word in words]
    return batch, words_to_spot

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

if __name__ == '__main__':
    batch_size = 2

    att_viz = AttentionVisualizer()
    device = get_device()

    dataset_types = [Data_Type.ELI5, Data_Type.STACK_EXCHANGE]
    knowledge_handler = load_kg_handler(kg_type=KGType.CONCEPTNET)
    tokenizer, model = load_tokenizer_and_model(knowledge_handler, device)

    training_data, validation_data, test_data = load_and_preprocess_data(dataset_types, limit=500)
    relations_metadata = get_kg_qa_data_metadata(knowledge_handler)
    relations_data = from_relations_path_2_relations(dataset_types, metadata=relations_metadata)
    dataset = RelationsDataset(training_data, relations_data=relations_data, tokenizer=tokenizer, device=device)

    batch, concepts_to_spot = input_configation(dataset)
    dataloader = DataLoader(
        batch, batch_size=batch_size, collate_fn=default_data_collator
    )
    examples_total, relations_total = get_examples(model, device, tokenizer, dataloader)
    plot_attn_lines_concepts_ids('concepts importance visualized',
                             examples_total[0],
                             10, 0,
                             concepts_to_spot,
                             relations_total[0])