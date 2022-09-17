
#############################
#   Imports
#############################

# Python modules
from typing import Optional, List

# Remote modules

# Local modules
from utils import read_txt_2_list, KGType
from metric_scorer import MetricScorer
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder
from kgs_binding.kg_qa_binding_utils import load_kg_handler

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class EvalGeneration:
    def __init__(self,
                 relation_mapper_builder:RelationsMapperBuilder,
                 sentences:Optional[List[str]]=None,
                 gold_sentences: Optional[List[str]]=None,
                 sentences_filename:Optional[str]=None):
        if sentences is None and sentences_filename is not None:
            sentences = read_txt_2_list(sentences_filename, 'eval/sentences')
        if gold_sentences is None and sentences_filename is not None:
            gold_sentences = read_txt_2_list('commongen.txt', 'eval/gold_sentences')
        self.sentences = sentences[:-1]
        self.gold_sentences = gold_sentences[:-1]
        #print('self.sentences[-2:]:', self.sentences[-2:])
        #print('self.gold_sentences[-2:]:', self.gold_sentences[-2:])
        self.metric_score = MetricScorer(relation_mapper_builder=relation_mapper_builder)

    def prep_blue_gold_sentences(self, labels_text):
        return [[label.strip()] for label in labels_text]

    def eval_sentences(self, sentences:Optional[List[str]]=None, gold_sentences:Optional[List[str]]=None):
        if not sentences:
            sentences = self.sentences
        if not gold_sentences:
            gold_sentences = self.gold_sentences
        blue_gold_sentences = self.prep_blue_gold_sentences(gold_sentences)
        #print('blue_gold_sentences[-10:]', blue_gold_sentences[-10:])
        metrics = self.metric_score.score_text_generation(sentences, gold_sentences, blue_gold_sentences)
        return metrics

"""
if __name__ == '__main__':
    kg = load_kg_handler(KGType.CONCEPTNET)
    relation_mapper_builder = RelationsMapperBuilder(kg)
    #sentences = ["the sky is red", "cats are afraid of dogs"]
    #gold_sentences = ["the sky is blue", "cats are good and afraid of dogs"]
    sentences_filename = "kg-bart_predictions_commongen.txt"
    ev = EvalGeneration(relation_mapper_builder=relation_mapper_builder, sentences_filename=sentences_filename)
    #results = ev.eval_sentences(sentences, gold_sentences)
    results = ev.eval_sentences()
    print(results)
"""