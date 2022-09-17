#############################
#   Imports
#############################

# Python modules

# Remote modules
from evaluate import load
import numpy as np

# Local modules
from eval.custom_metric import compute_bleu_score, Cider, Spice, calc_accuracy
from data.relation_utils import clean_relations
from eval.bleu_metric import Bleu
import spacy

#############################
#   Constants
#############################
bleu_metric = load('bleu')
sacrebleu_metric = load('sacrebleu')
meteor_metric = load('meteor')
rouge_metric = load('rouge')
accuracy_metric = load("accuracy")
spice = Spice()
cider = Cider()
blue = Bleu(4)
nlp = spacy.load("en_core_web_sm")

def tokenize(sentences):
    new_sentences = []
    for sentence in sentences:
        a = ''
        for token in nlp(sentence):
            a += token.text
            a += ' '
        new_sentences.append(a.rstrip())
    return new_sentences

#############################
#   Stuff
#############################
class MetricScorer:

    def __init__(self, relation_mapper_builder):
        self.relation_mapper_builder = relation_mapper_builder

    def score_commonsense_relations(self, contexts):
        commonsense_relations_info = self.relation_mapper_builder.get_relations_mapping_complex(context=contexts, clear_common_wds=True)
        # clean relation
        commonsense_relations = clean_relations(commonsense_relations_info)
        counter = 0
        for sample in commonsense_relations:
            for concept, outgoing_rels in sample.items():
                counter += len(outgoing_rels.values())
        return counter / len(commonsense_relations)

    def score_relative_commonsense_relations(self, number_of_relations_pred, number_of_relations_results_gold):
        return number_of_relations_pred / number_of_relations_results_gold

    def score_coverage(self, coverage_batch):
        coverage_batch_sum = [len(individual_coverage) for individual_coverage in coverage_batch]
        return sum(coverage_batch_sum)/len(coverage_batch)

    def get_word_count_of_phrases(self, sentences):
        return [len(sentence.split(' ')) for sentence in sentences]

    def score_size(self, preds, gold_refs):
        len_preds = np.mean(self.get_word_count_of_phrases(preds))
        len_gold_refs = np.mean(self.get_word_count_of_phrases(gold_refs))
        return len_preds, len_preds/len_gold_refs

    def preprocess_text_generation(self, tokenizer, preds, labels):
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds_str = tokenizer.batch_decode(preds, skip_special_tokens=False)
        print('test preds:', decoded_preds_str[0])

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        blue_decoded_labels = [[label.strip()] for label in decoded_labels]
        decoded_labels = [label.strip() for label in decoded_labels]
        return decoded_preds, decoded_labels, blue_decoded_labels

    def score_text_generation(self, decoded_preds, decoded_labels, blue_decoded_labels):
        bleu_result_f = compute_bleu_score(bleu_metric, decoded_preds, blue_decoded_labels, max_order=3, smooth=True)
        bleu_result = blue.compute_score(tokenize(decoded_preds), tokenize(decoded_labels))
        try:
            sacrebleu_result = sacrebleu_metric.compute(predictions=decoded_preds, references=blue_decoded_labels)
        except Exception as _:
            sacrebleu_result = {'score': bleu_result_f["bleu"] * 100.0}
        meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=False)
        rouge_results = rouge_results['rougeL']
        rouge_results = np.mean([s.fmeasure for s in rouge_results])
        number_of_relations_results = self.score_commonsense_relations(decoded_preds)
        gold_number_of_relations_results = self.score_commonsense_relations(decoded_labels)
        relative_relations_results = self.score_relative_commonsense_relations(number_of_relations_results, gold_number_of_relations_results)
        coverage_words_pred = self.relation_mapper_builder.get_kg_concepts_from_context(decoded_preds, clear_common_wds=True)
        coverage_words_gold = self.relation_mapper_builder.get_kg_concepts_from_context(decoded_labels, clear_common_wds=True)
        #print('coverage_words_pred:', coverage_words_pred)
        coverage = self.score_coverage(coverage_words_pred)
        coverage_relative = coverage / self.score_coverage(coverage_words_gold)
        cider_results = cider.compute_score(decoded_preds, decoded_labels)[0] #get average_score which is at index 0
        spice_results = spice.compute_score(decoded_preds, decoded_labels)
        len_results, len_relative_results = self.score_size(decoded_preds, decoded_labels)
        cov_vs_len = coverage/len_results
        metrics = {
            "bleu_f": bleu_result_f["bleu"] * 100.0,
            "bleu": bleu_result[0],
            "sacrebleu": sacrebleu_result["score"], #already scaled to 100
            "meteor": meteor_result["meteor"] * 100.0,
            "rouge": rouge_results * 100.0,
            "CIDEr": cider_results * 10.0, # because for some weird reasons its maxed to 10.0
            "SPICE": spice_results * 100.0,
            "relations_weight": number_of_relations_results,
            "relations_weight_relative": relative_relations_results * 100.0,
            "coverage": coverage,
            "coverage_relative": coverage_relative * 100.0,
            "len": len_results,
            "len_relative": len_relative_results * 100.0,
            "cov_vs_len": cov_vs_len * 100.0
        }
        metrics = {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}
        metrics["combined"] = (metrics["sacrebleu"] + metrics["meteor"] + metrics["rouge"]) / 3
        return metrics

    def prepare_csqa_text_generation(self, tokenizer, simple_bart_tokenizer, preds, labels):
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        print("preds_first: ", preds[0])
        preds = np.roll(preds, -1, axis=1)
        preds = preds.transpose()
        preds[:][-1] = 1
        preds = preds.transpose()
        decoded_preds = simple_bart_tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = simple_bart_tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("predictions: ", decoded_preds[0])
        print("references: ", decoded_labels[0])

        new_preds = simple_bart_tokenizer(decoded_preds, add_special_tokens=False).input_ids
        new_labels = simple_bart_tokenizer(decoded_labels, add_special_tokens=False).input_ids

        print("predictions_ids: ", new_preds[0])
        print("references_ids: ", new_labels[0])

        return new_preds, new_labels, decoded_preds, decoded_labels

    def score_text_csqa(self, preds, labels):
        accuracy_vec = [1 if pred.strip() == lab.strip() else 0 for pred,lab in zip(preds, labels)]
        accuracy_score = sum(accuracy_vec) / len(preds)
        metrics = {
            "accuracy": accuracy_score * 100.0,
        }
        return metrics

    def score_csqa(self, preds, labels):
        accuracy_result = calc_accuracy(accuracy_metric=accuracy_metric, predictions=preds, references=labels)
        metrics = {
            "accuracy": accuracy_result["accuracy"],
        }
        return metrics