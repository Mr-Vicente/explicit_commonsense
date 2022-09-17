#############################
#   Imports
#############################

# Python modules

# Remote modules
import torch
import numpy as np
from transformers import LogitsProcessor

# Local modules
from utils import ScoringType
from kgs_binding.parsing_utils import ParsingUtils

#############################
#   Constants
#############################

#############################
#   Stuff
#############################
class KGLogitsProcessor(LogitsProcessor):
  def __init__(self, tokenizer, relations_mapper,
               top_n=50, scoring_type:ScoringType=ScoringType.DEFAULT,
               ignore_common_words=True):
      self.tokenizer = tokenizer
      self.relations_mapper = relations_mapper
      self.top_n = top_n
      self.scoring_type=scoring_type
      self.ignore_common_words=ignore_common_words

  def __call__(self, input_ids, scores):
    # for every beam (partially generated sentence)
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
      # get the last token of this beam
      kg_nodes_counter=0
      current_phrase = self.tokenizer.decode(beam_input_ids)
      #print('current_phrase:', current_phrase)
      current_concepts = self.relations_mapper.get_kg_concepts_from_context([current_phrase], clear_common_wds=True)[0]
      #print('current_concepts:', current_concepts)
      for current_concept in current_concepts:
        current_concept_norm = self.relations_mapper.knowledge.normalize_nouns(current_concept)
        indeces_to_use = torch.argsort(beam_scores, dim=0, descending=True)[:self.top_n]
        #print('current_concept_norm:', current_concept_norm)
        #print('indeces_to_use:', indeces_to_use[:2])
        #t_indeces_to_use represent the top ids which should come next
        for top_id in indeces_to_use:
          next_word_norm = self.tokenizer.decode(top_id).strip()  # TODO
          #print('next_word_norm:', next_word_norm)
          if self.relations_mapper.knowledge.exists_relation_between(current_concept_norm, next_word_norm)\
                  and ParsingUtils.is_word_a_relevant_one(self.ignore_common_words, next_word_norm):
            prev_score = beam_scores[top_id]
            #print('prev_score:', prev_score)
            #print('next_word_norm:', next_word_norm)
            scores[beam_index, top_id] = self.set_score(beam_scores, indeces_to_use, prev_score, kg_nodes_counter)
            kg_nodes_counter+=1
    return scores

  def set_score(self, beam_scores, indeces_to_use, next_token_curr_score, kg_nodes_counter):
    if self.scoring_type == ScoringType.DEFAULT:
      return next_token_curr_score # dont change score
    elif self.scoring_type == ScoringType.MAX_PROB:
      max_logit_score = beam_scores[indeces_to_use[0]] # equivlent to max(beam_scores)
      return max_logit_score
    elif self.scoring_type == ScoringType.INTERPOL:
      start_idx, end_idx = 0, 9
      max_logit_score = beam_scores[indeces_to_use[start_idx]]  # equivlent to max(beam_scores)
      top10_logit_score = beam_scores[indeces_to_use[end_idx]]  # equivlent to max(beam_scores)
      x = [start_idx, max_logit_score]
      y = [end_idx, top10_logit_score]
      if kg_nodes_counter > end_idx:
        return next_token_curr_score
      logit_correspondence = np.interp(kg_nodes_counter, x,y)
      return logit_correspondence
    else:
      raise NotImplementedError()

class kg_hop1_Processor(LogitsProcessor):
  def __init__(self, tokenizer):
      self.tokenizer = tokenizer

  def __call__(self, input_ids, scores, **kwargs):
    # for every beam (partially generated sentence)
    choices_ids = kwargs['choices_ids']
    vocab_indexes = torch.arange(self.tokenizer.vocab_size)
    vocab_size = len(vocab_indexes)
    for beam_index, (_beam_input_ids, _beam_scores) in enumerate(zip(input_ids, scores)):
      example_id = beam_index // len(choices_ids)
      valid_ids_example = choices_ids[example_id]
      beam_score_temp = torch.ones(vocab_size).to(input_ids.device)
      beam_score_temp[valid_ids_example] = 0
      beam_score_temp = beam_score_temp.masked_fill(beam_score_temp == 1, -float("inf"))
      beam_score_temp[valid_ids_example] = scores[beam_index][valid_ids_example]
      scores[beam_index] = beam_score_temp
      del beam_score_temp
    return scores

class MultipleChoiceProcessor(LogitsProcessor):
  def __init__(self, tokenizer, choices_ids):
      self.tokenizer = tokenizer
      self.choices_ids = choices_ids
      #print('choices_ids_size:', len(choices_ids))
      #print('choices_ids:', choices_ids)

  def __call__(self, input_ids, scores):
    # for every beam (partially generated sentence)
    #choices_ids = kwargs['choices_ids']
    vocab_indexes = torch.arange(self.tokenizer.vocab_size)
    vocab_size = len(vocab_indexes)
    #print('input_ids_size:', input_ids.shape)
    #print('input_ids:', input_ids)
    for beam_index, (_beam_input_ids, _beam_scores) in enumerate(zip(input_ids, scores)):
      example_id = int(beam_index // (input_ids.shape[0]//len(self.choices_ids)))
      valid_ids_example = torch.cat((self.choices_ids[example_id],torch.tensor([
        self.tokenizer.bos_token_id,
        self.tokenizer.pad_token_id,
        self.tokenizer.eos_token_id,
      ]).to(input_ids.device)))
      beam_score_temp = torch.ones(vocab_size).to(input_ids.device)
      beam_score_temp[valid_ids_example] = 0
      beam_score_temp = beam_score_temp.masked_fill(beam_score_temp == 1, -float("inf"))
      beam_score_temp[valid_ids_example] = scores[beam_index][valid_ids_example]
      scores[beam_index] = beam_score_temp
      del beam_score_temp
    return scores