#############################
#   Imports
#############################

# Python modules

# Remote modules
import torch

# Local modules
from data.relation_utils import clean_relations, tokens_with_relations

from trainer.loss_scores import (
    score_based_on_amount_of_relations,
    score_based_on_lack_of_relational_concepts,
    score_based_on_matches,
    score_on_concept_richness
)

#############################
#   Constants
#############################

#############################
#   Helper funcs
#############################
def get_original_rels_amount(input_data):
    input_commonsense = input_data.get('input_commonsense_relations')
    #print('input_commonsense.shape:', input_commonsense.shape) #[128, 32, 32]
    input_commonsense[input_commonsense > 1] = 1
    rels_amount = input_commonsense.sum(dim=(1,2))
    return rels_amount

def get_concepts_with_no_relations(relation_mapper_builder, phrase_generated):
    #print('phrase_generated:', phrase_generated)
    #all_concepts = self.relation_mapper_builder.get_kg_concepts_from_context([phrase_generated], clear_common_wds=True)[0]
    all_concepts = phrase_generated.strip().split(' ') # all words
    concepts_with_relations = relation_mapper_builder.get_concepts_from_context(phrase_generated, clear_common_wds=True)
    #print('all_concepts:', all_concepts)
    #print('concepts_with_relations:', concepts_with_relations)
    concepts_with_no_relations = list(set(all_concepts).difference(concepts_with_relations))
    #print('without_relations:', concepts_with_no_relations)
    return concepts_with_no_relations, concepts_with_relations, all_concepts

def get_pred_and_label_concepts(relation_mapper_builder, gen_phrase, label_phrase):
    gen_concepts = relation_mapper_builder.get_kg_concepts_from_context([gen_phrase], clear_common_wds=True)[0]
    label_concepts = relation_mapper_builder.get_kg_concepts_from_context([label_phrase], clear_common_wds=True)[0]
    return gen_concepts, label_concepts

def process_context(relation_mapper_builder, context):
    # process context in search for relations
    commonsense_relations = relation_mapper_builder.get_relations_mapping_complex(context=[context])
    # clean relation
    commonsense_relation = clean_relations(commonsense_relations)[0]
    #print('loss context:', context)
    #print('loss phrase:', commonsense_relation)
    return commonsense_relation

#############################
#   Losses
#############################

def calc_cp_def_loss(def_loss_function, inputs, model, return_outputs=True, alpha=1.0):
    input_commonsense = inputs.get('input_commonsense_relations')
    input_ids = inputs.get('input_ids')
    inputs['reduce_ce'] = False
    loss, outputs = def_loss_function(model, inputs, return_outputs=return_outputs)
    logits = outputs.logits
    masked_index = input_ids >= 3  # ignore special_tokens
    penalize_batch_preds = []
    relations_1d = tokens_with_relations(input_commonsense)
    relevant_ids = input_ids[relations_1d]
    # relevant_ids = relevant_ids[masked_index]
    for b, b_logs in enumerate(logits):
        probs = b_logs[masked_index[b]].softmax(dim=0)
        _values, predictions = probs.topk(1)
        score = score_based_on_matches(relevant_ids, predictions)
        penalize_batch_preds.append(score)
    # Resize and average loss per sample
    # print('loss:', loss)
    # print('loss.shape:', loss.shape)
    loss_per_sample = loss.view(logits.size(0), logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    weights = torch.tensor(penalize_batch_preds).to(input_ids.device)
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    loss = (loss_per_sample * weights).mean()
    return (loss, outputs) if return_outputs else loss

def calc_prp_nrp_def_loss(def_loss_function, inputs, model, tokenizer,
                        relation_mapper_builder, return_outputs=True):
    original_rels_amount = get_original_rels_amount(inputs)
    # print('original_rels_amount:', original_rels_amount)
    input_ids = inputs.get('input_ids')
    #inputs['reduce_ce'] = False
    loss, outputs = def_loss_function(model, inputs, return_outputs=return_outputs)
    logits = outputs.logits
    masked_index = input_ids >= 3  # ignore special_tokens
    penalize_batch_preds = []
    penalize_batch_no_relations = []
    # relevant_ids = relevant_ids[masked_index]
    # print('logits.shape:', logits.shape)
    for b, b_logs in enumerate(logits):
        probs = b_logs[masked_index[b]].softmax(dim=0)
        _values, predictions = probs.topk(1)
        gen_phrase = tokenizer.decode(predictions.squeeze(1), skip_special_tokens=True)
        gen_phrase = gen_phrase.strip()
        # print('gen_phrase:', gen_phrase)
        reals_info = process_context(relation_mapper_builder, gen_phrase)
        concepts_with_no_relations, concepts_with_relations, all_concepts = get_concepts_with_no_relations(relation_mapper_builder, gen_phrase)
        all_words = gen_phrase.strip().split(' ')
        score_concepts_with_no_rel = score_based_on_lack_of_relational_concepts(all_words, concepts_with_no_relations)
        score = score_based_on_amount_of_relations(reals_info, original_rels_amount=original_rels_amount[b])
        del reals_info
        penalize_batch_preds.append(score)
        penalize_batch_no_relations.append(score_concepts_with_no_rel)

    scoring_relations_weights = torch.tensor(penalize_batch_preds).to(input_ids.device)
    scoring_relations_weights = scoring_relations_weights.mean()
    penalize_no_relations = torch.tensor(penalize_batch_no_relations).to(input_ids.device)
    penalize_no_relations = penalize_no_relations.mean()

    loss = loss + scoring_relations_weights + penalize_no_relations
    return (loss, outputs) if return_outputs else loss

def calc_cp_rp_def_loss(def_loss_function, inputs, model, tokenizer,
                        relation_mapper_builder, return_outputs=True, alpha=1.0):
    #input_commonsense = inputs.get('input_commonsense_relations')
    # print('original_rels_amount:', original_rels_amount)
    input_ids = inputs.get('input_ids')
    labels_ids = inputs.get('labels')
    inputs['reduce_ce'] = False
    loss, outputs = def_loss_function(model, inputs, return_outputs=return_outputs)
    logits = outputs.logits
    masked_index = input_ids >= 3  # ignore special_tokens
    penalize_lack_of_concepts = []
    penalize_batch_no_relations = []
    # relevant_ids = relevant_ids[masked_index]
    # print('logits.shape:', logits.shape)
    for b, b_logs in enumerate(logits):
        probs = b_logs[masked_index[b]].softmax(dim=0)
        _values, predictions = probs.topk(1)

        #relations_1d = tokens_with_relations(input_commonsense[b])
        #relevant_ids = input_ids[b][relations_1d]

        preds_lower_tensor = predictions.squeeze(1)
        gen_phrase = tokenizer.decode(preds_lower_tensor, skip_special_tokens=True)
        label_phrase = tokenizer.decode(labels_ids[b], skip_special_tokens=True)
        #concept_score = score_based_on_matches(relevant_ids, preds_lower_tensor)
        gen_phrase = gen_phrase.strip()
        #print('gen_phrase:', gen_phrase)
        concepts_with_no_relations, concepts_with_relations, all_concepts = get_concepts_with_no_relations(relation_mapper_builder, gen_phrase)
        all_words = gen_phrase.split(' ')
        label_of_relations = relation_mapper_builder.get_concepts_from_context(label_phrase, clear_common_wds=True)
        gen_concepts, label_concepts = get_pred_and_label_concepts(relation_mapper_builder, gen_phrase, label_phrase)
        concept_score = score_on_concept_richness(gen_concepts, label_concepts, seq_len=len(input_ids))
        #print('label_of_relations:', label_of_relations)
        label_amount_of_relations = len(label_of_relations)
        score_concepts_with_no_rel = score_based_on_lack_of_relational_concepts(all_words, concepts_with_no_relations, label_amount_of_relations, seq_len=len(input_ids))
        penalize_lack_of_concepts.append(concept_score)
        penalize_batch_no_relations.append(score_concepts_with_no_rel)
        #print('concept_score:', concept_score)
        #print('score_concepts_with_no_rel_score:', score_concepts_with_no_rel)

    scoring_lack_of_concepts = torch.tensor(penalize_lack_of_concepts).to(input_ids.device)

    penalize_no_relations = torch.tensor(penalize_batch_no_relations).to(input_ids.device)
    #penalize_no_relations = penalize_no_relations.mean()

    loss_per_sample = loss.view(logits.size(0), logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    #concepts_weights = alpha * (1.0 + scoring_lack_of_concepts)
    #no_relations_weights = alpha * (1.0 + penalize_no_relations)
    concepts_weights = scoring_lack_of_concepts
    no_relations_weights = penalize_no_relations

    #loss = (loss_per_sample * concepts_weights).mean() + (loss_per_sample * no_relations_weights).mean()
    loss = (loss_per_sample + concepts_weights + no_relations_weights).mean()
    return (loss, outputs) if return_outputs else loss

def calc_def_constraint_loss(def_loss_function, inputs, model, tokenizer,return_outputs=True, alpha=1.0):
    # print('original_rels_amount:', original_rels_amount)
    input_ids = inputs.get('input_ids')
    inputs['reduce_ce'] = False
    loss, outputs = def_loss_function(model, inputs, return_outputs=return_outputs)
    logits = outputs.logits
    masked_index = input_ids >= 3  # ignore special_tokens

    penalise_multiple_sentences=[]
    for b, b_logs in enumerate(logits):
        probs = b_logs[masked_index[b]].softmax(dim=0)
        _values, predictions = probs.topk(1)
        preds_lower_tensor = predictions.squeeze(1)
        gen_phrase = tokenizer.decode(preds_lower_tensor, skip_special_tokens=True)
        gen_phrase = gen_phrase.strip()
        next_phrase = gen_phrase.split('.')
        if len(next_phrase)>1:
            penalty=1
        else:
            penalty=0
        penalise_multiple_sentences.append(penalty)

    penalise_multiple_sentences = torch.tensor(penalise_multiple_sentences).to(input_ids.device)
    loss_per_sample = loss.view(logits.size(0), logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    multi_sents_weights = alpha * (1.0 + penalise_multiple_sentences)
    #no_relations_weights = alpha * (1.0 + penalize_no_relations)

    #loss = (loss_per_sample * concepts_weights).mean() + (loss_per_sample * no_relations_weights).mean()
    loss = (loss_per_sample + multi_sents_weights).mean()
    return (loss, outputs) if return_outputs else loss