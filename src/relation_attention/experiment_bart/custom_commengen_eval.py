
#############################
#   Imports
#############################

# Python modules
from typing import Optional, List

# Remote modules
import pandas
import csv

# Local modules
from utils import read_txt_2_list, KGType, read_json_file_2_dict
from eval.metric_scorer import MetricScorer
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder
from kgs_binding.kg_qa_binding_utils import load_kg_handler


def prep_blue_gold_sentences(labels_text):
    return [[label.strip()] for label in labels_text]


def eval_sentences(metric_scorer, sentences: Optional[List[str]] = None, gold_sentences: Optional[List[str]] = None):
    blue_gold_sentences = prep_blue_gold_sentences(gold_sentences)
    # print('blue_gold_sentences[-10:]', blue_gold_sentences[-10:])
    metrics = metric_scorer.score_text_generation(sentences, gold_sentences, blue_gold_sentences)
    return metrics

if __name__ == '__main__':
    kg = load_kg_handler(KGType.CONCEPTNET)
    relation_mapper_builder = RelationsMapperBuilder(kg)
    metric_scorer = MetricScorer(relation_mapper_builder=relation_mapper_builder)

    df_input = pandas.read_csv('eval/human_data/human_data_complete.csv', delimiter=',')
    results = read_json_file_2_dict('human_sentences.json', 'eval/human_data')
    df_results = pandas.DataFrame(results)
    #print(df_input)
    #print(df_results)

    #temp = df_input.merge(df_results,how='inner',on=['concepts_input'])
    #print(temp[temp['_merge'] == 'left_only'])
    #print(temp[temp['_merge'] == 'right_only'])


    #with open('eval/human_data/human_data_complete.csv', 'w') as f:
    #    human_writer = csv.writer(f)
    #    human_writer.writerow([*df_input.columns.values.tolist(), 'human_sentences'])
    #    inputs = list(df_input['concepts_input'])
    #    golds = list(df_input['gold_sentence'])
    #    human_sentences = list(temp['human_sentences'])
    #    for inp, gold, human_sentence in zip(inputs, golds, human_sentences):
    #        human_writer.writerow([inp, gold, human_sentence])


    #sentences = list(df_results['commonsense_model_4'])
    sentences = list(df_input['human_sentences'])
    gold_sentences = list(df_input['gold_sentence'])
    print(sentences[0], gold_sentences[0])
    human = eval_sentences(metric_scorer, sentences, gold_sentences)
    print(human)
