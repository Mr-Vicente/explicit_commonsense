#############################
#   Imports
#############################

# Python modules
from typing import List
import itertools
from operator import itemgetter

# Remote modules
import numpy as np
import torch
#from torchtext.vocab import GloVe

# Local modules
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder
from data.relation_utils import clean_relations
from model_utils import create_layers_head_mask

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
    DisjunctiveConstraint,
)

from utils import get_jump_chunks

#############################
#   Constants
#############################

#############################
#   Stuff
#############################
from custom_tokenizer import BartCustomTokenizerFast
from custom_bart import BartCustomConfig, BartCustomForConditionalGeneration
from utils import get_device, KGType, Model_Type

from kgs_binding.kg_base_wrapper import KGBaseHandler
from kgs_binding.cskg_handler import CSKGHandler
from kgs_binding.swow_handler import SwowHandler
from kgs_binding.conceptnet_handler import ConceptNetHandler

class Inference:
    def __init__(self, model_path:str, max_length=32):
        self.device = get_device()
        self.tokenizer = self.prepare_tokenizer()
        self.model = self.prepare_model(model_path)
        self.max_length = max_length

    def prepare_tokenizer(self):
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        return tokenizer

    def prepare_model(self, model_path):
        config = BartConfig.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path, config=config).to(self.device)
        model.eval()
        return model

    def pre_process_context(self, context):
        context = context.lower()
        context_tokenized = self.tokenizer(context, padding='max_length',
                                truncation='longest_first',  max_length=self.max_length,
                                return_tensors="pt",
                                )
        return context_tokenized

    def generate_based_on_context(self, context):
        model_input = self.pre_process_context(context)
        generated_answers_encoded = self.model.generate(input_ids=model_input["input_ids"].to(self.device),
                                                   attention_mask=model_input["attention_mask"].to(self.device),
                                                   min_length=1,
                                                   max_length=self.max_length,
                                                   do_sample=True,
                                                   early_stopping=True,
                                                   num_beams=4,
                                                   temperature=1.0,
                                                   top_k=None,
                                                   top_p=None,
                                                   # eos_token_id=tokenizer.eos_token_id,
                                                   no_repeat_ngram_size=2,
                                                   num_return_sequences=1,
                                                   return_dict_in_generate=True,
                                                   output_attentions=True,
                                                   output_scores=True)
        # print(f'Scores: {generated_answers_encoded}')
        response = self.tokenizer.batch_decode(generated_answers_encoded['sequences'], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
        encoder_attentions = generated_answers_encoded['encoder_attentions']
        return response, encoder_attentions, model_input

    def prepare_context_for_visualization(self, context):
        examples = []
        response, encoder_outputs, model_input = self.generate_based_on_context(context)
        encoder_outputs = torch.stack(encoder_outputs)
        n_layers, batch_size, n_heads, src, tgt = encoder_outputs.size()
        print(encoder_outputs.size())
        encoder_attentions = encoder_outputs.view(batch_size, n_layers, n_heads, src, tgt)
        for i, ex in enumerate(encoder_attentions):
            d = {}
            indices = model_input['input_ids'][i].detach().cpu()
            all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
            useful_indeces = indices != self.tokenizer.pad_token_id
            all_tokens = np.array(all_tokens)[useful_indeces]
            all_tokens = [tok.replace('Ġ', '') for tok in all_tokens]
            d['words'] = all_tokens
            d['attentions'] = ex.detach().cpu().numpy()
            examples.append(d)
        print(d['words'])
        return response, examples

class RelationsInference:
    def __init__(self, model_path:str, kg_type: KGType, model_type:Model_Type, max_length=32):
        self.device = get_device()
        kg_handler: KGBaseHandler = self.select_kg(kg_type)
        self.kg_handler = kg_handler
        relation_names = kg_handler.get_relation_types()
        self.tokenizer = self.prepare_tokenizer(relation_names, model_type)
        self.simple_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.model, self.config = self.prepare_model(relation_names, model_path, model_type)
        self.relation_mapper_builder = RelationsMapperBuilder(knowledge=kg_handler)
        self.max_length = max_length
        #self.glove = GloVe(name='6B', dim=50)

    def prepare_tokenizer(self, relation_names: List[str], model_type:Model_Type):
        tokenizer = BartCustomTokenizerFast.from_pretrained('facebook/bart-large')
        tokenizer.set_known_relation_names(relation_names)
        tokenizer.set_operation_mode(there_is_difference_between_relations=model_type.there_is_difference_between_relations())
        return tokenizer

    def prepare_model(self, relation_names: List[str], model_path, model_type:Model_Type):
        config = BartCustomConfig.from_pretrained(model_path)
        print('config.heads_mask:', config.heads_mask)
        if config.num_relation_kinds is None:
            config.num_relation_kinds = len(relation_names)
        if config.is_simple_mask_commonsense is None:
            config.is_simple_mask_commonsense = model_type.is_simple_mask_commonsense()
        if config.heads_mask is None:
            config.heads_mask = create_layers_head_mask(config)#, heads_mask_type, specific_heads)
        model = BartCustomForConditionalGeneration.from_pretrained(model_path, config=config).to(self.device)
        model.eval()
        return model, config

    def select_kg(self, kg_type: KGType = KGType.SWOW):
        if kg_type.value == KGType.SWOW.value:
            return SwowHandler()
        elif kg_type.value == KGType.CSKG.value:
            return CSKGHandler()
        elif kg_type.value == KGType.CONCEPTNET.value:
            return ConceptNetHandler()
        else:
            raise NotImplementedError()

    def pre_process_context(self, context):
        context = context.lower()
        # process context in search for relations
        commonsense_relations = self.relation_mapper_builder.get_relations_mapping_complex(context=[context], clear_common_wds=True)
        # clean relation
        commonsense_relation = clean_relations(commonsense_relations)[0]
        # convert this relations to matrices
        print(commonsense_relation)
        context_tokenized = self.tokenizer(context, padding='max_length',
                                truncation='longest_first',  max_length=self.max_length,
                                return_tensors="pt", return_offsets_mapping=True,
                                input_commonsense_relations=commonsense_relation,
        )
        return context_tokenized

    def get_relations_information(self, phrase_generated):
        #print('phrase_generated:', phrase_generated)
        all_concepts = self.relation_mapper_builder.get_kg_concepts_from_context([phrase_generated], clear_common_wds=True)[0]
        words = phrase_generated.strip().split(' ') # all words
        concepts_with_relations = self.relation_mapper_builder.get_concepts_from_context(phrase_generated, clear_common_wds=True)
        #print('all_concepts:', all_concepts)
        #print('concepts_with_relations:', concepts_with_relations)
        concepts_with_no_relations = list(set(all_concepts).difference(concepts_with_relations))
        #print('without_relations:', concepts_with_no_relations)
        print("====== RELATIONS SUMMARY ======")
        print('phrase_generated:', phrase_generated)
        print('words:', words)
        print('all_concepts:', all_concepts)
        print('concepts_with_relations:', concepts_with_relations)
        print('without_relations:', concepts_with_no_relations)
        print("\n== STATS:")
        print('n_words:', len(words))
        print('n_concepts:', len(all_concepts))
        print('n_concepts_with_relations:', len(concepts_with_relations))
        print('n_c_without_relations:', len(concepts_with_no_relations))
        print("====== ================= ======")
        return words, all_concepts, concepts_with_relations, concepts_with_no_relations

    def most_similar_words(self, reference, other):
        data = []
        for w in other:
            dist = torch.norm(self.glove[reference] - self.glove[w]).item()  # euclidean distance
            # print(w, float(dist))
            #n_words = len(w.split(' '))
            data.append((w, dist))
        sorted_data = sorted(data, key=itemgetter(1), reverse=False)
        simple_sorted_data = [w for w, score in sorted_data]
        return simple_sorted_data

    def remove_subsets(self, l):
        #l = [[1, 2, 4, 8], [1, 2, 4, 5, 6], [1, 2, 3], [2, 3, 21], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7]]
        l2 = l[:]
        for m in l:
            for n in l:
                if set(m).issubset(set(n)) and m != n:
                    l2.remove(m)
                    break
        return l2

    def generate_based_on_context(self, context, use_kg=False):
        model_input = self.pre_process_context(context)
        #print(model_input)
        gen_kwargs = {}
        if "input_commonsense_relations" in model_input:
            #print(model_input['input_commonsense_relations'].sum())
            gen_kwargs["relation_inputs"] = model_input.get("input_commonsense_relations").to(self.device)

        constraints = None
        if use_kg:
            constraints = []
            concepts_from_context = self.relation_mapper_builder.get_concepts_from_context(context=context, clear_common_wds=True)
            useful_concepts = [self.relation_mapper_builder.knowledge.get_related_concepts(concept) for concept in concepts_from_context]
            if not useful_concepts:
                useful_concepts = [self.kg_handler.get_related_concepts(concept) for concept in concepts_from_context]
            useful_concepts = [[f'{phrase}' for phrase in concepts] for concepts in useful_concepts] # add spaces
            #useful_concepts = [[phrase for phrase in concepts if len(phrase.split(' ')) == 1] for concepts in useful_concepts]
            #useful_concepts = list(itertools.chain.from_iterable(useful_concepts))
            #print('useful_concepts:', useful_concepts[:5])
            if concepts_from_context:
                for context_concept, neighbour_concepts in zip(concepts_from_context, useful_concepts):
                    print('neighbour:', neighbour_concepts[:20])
                    #flexible_words = self.most_similar_words(context_concept, neighbour_concepts) # limit the upperbound
                    #flexible_words = [word for word in flexible_words if word not in context_concept] # remove input concepts
                    flexible_words = [word for word in neighbour_concepts if word not in context_concept]  # remove input concepts
                    flexible_words_ids: List[List[int]] = self.simple_tokenizer(flexible_words, add_prefix_space=True,add_special_tokens=False).input_ids
                    flexible_words_ids = self.remove_subsets(flexible_words_ids)
                    #add_prefix_space=True
                    #flexible_words_ids = [x for x in flexible_words_ids if len(x) == 1] # problem with subsets
                    flexible_words_ids = flexible_words_ids[:10]
                    print('flexible_words_ids:', flexible_words_ids[:3])
                    constraint = DisjunctiveConstraint(flexible_words_ids)
                    constraints.append(constraint)
            else:
                constraints = None

        generated_answers_encoded = self.model.generate(input_ids=model_input["input_ids"].to(self.device),
                                                   attention_mask=model_input["attention_mask"].to(self.device),
                                                   constraints=constraints,
                                                   min_length=1,
                                                   max_length=self.max_length,
                                                   do_sample=False,
                                                   early_stopping=True,
                                                   num_beams=8,
                                                   temperature=1.0,
                                                   top_k=None,
                                                   top_p=None,
                                                   # eos_token_id=tokenizer.eos_token_id,
                                                   no_repeat_ngram_size=2,
                                                   num_return_sequences=1,
                                                   return_dict_in_generate=True,
                                                   output_attentions=True,
                                                   output_scores=True,
                                                   **gen_kwargs,
                                                   )
        # print(f'Scores: {generated_answers_encoded}')
        response = self.tokenizer.batch_decode(generated_answers_encoded['sequences'], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
        encoder_attentions = generated_answers_encoded['encoder_attentions']
        return response, encoder_attentions, model_input

    def get_related_concepts_list(self, knowledge, list_concepts):
        other_concepts = []
        for concept in list_concepts:
            other_near_concepts = knowledge.get_related_concepts(concept)
            other_concepts.extend(other_near_concepts)
        return other_concepts


    def generate_contrained_based_on_context(self, contexts, use_kg=True, max_concepts=1):
        model_inputs = [self.pre_process_context(context) for context in contexts]
        constraints = None
        if use_kg:
            constraints = []
            concepts_from_contexts = [self.relation_mapper_builder.get_concepts_from_context(context=context, clear_common_wds=True) for context in contexts]
            neighbours_contexts = []#[self.get_related_concepts_list(self.relation_mapper_builder.knowledge, context) for context in concepts_from_contexts]
            if not neighbours_contexts:
                neighbours_contexts = [self.get_related_concepts_list(self.kg_handler, context) for context in concepts_from_contexts]
            all_constraints = []
            for context_neighbours in neighbours_contexts:
                # context_neighbours is a collection of concepts
                # lets create sub collections of concepts
                context_neighbours = [f' {concept}' for concept in context_neighbours if len(concept) > 3]
                n_size_chuncks = len(context_neighbours) // max_concepts
                n_size_chuncks = n_size_chuncks if n_size_chuncks > 0 else 1
                sub_concepts_collection = list(get_jump_chunks(context_neighbours, jump=n_size_chuncks))
                constraints = []
                for sub_concepts in sub_concepts_collection[:max_concepts]:
                    flexible_words_ids: List[List[int]] = self.tokenizer(sub_concepts,
                                                                    add_special_tokens=False).input_ids  # add_prefix_space=True,
                    # flexible_words_ids = self.remove_subsets(flexible_words_ids)
                    flexible_words_ids = [[word_ids[0]] for word_ids in flexible_words_ids]
                    disjunctive_set = list(map(list, set(map(frozenset, flexible_words_ids))))
                    if not any(disjunctive_set):
                        continue
                    constraint = DisjunctiveConstraint(disjunctive_set)
                    constraints.append(constraint)
                if not any(constraints):
                    constraints = None
                all_constraints.append(constraints)
        else:
            all_constraints = None
        if not all_constraints:
            all_constraints = None

        generated_answers_encoded = []
        encoder_attentions_list = []
        for i, contraints in enumerate(all_constraints):
            #print('contraints.token_ids:', [x.token_ids for x in contraints])
            gen_kwargs = {}
            inputs = model_inputs[i]
            if "input_commonsense_relations" in inputs:
                # print(model_input['input_commonsense_relations'].sum())
                gen_kwargs["relation_inputs"] = inputs.get("input_commonsense_relations").to(self.device)
            #print('model_kwargs.get("attention_mask"):', model_kwargs.get("attention_mask"))
            gen = self.model.generate(input_ids=inputs["input_ids"].to(self.device),
                               attention_mask=inputs["attention_mask"].to(self.device),
                               constraints=constraints,
                               min_length=1,
                               max_length=self.max_length,
                               do_sample=False,
                               early_stopping=True,
                               num_beams=8,
                               temperature=1.0,
                               top_k=None,
                               top_p=None,
                               # eos_token_id=tokenizer.eos_token_id,
                               no_repeat_ngram_size=2,
                               num_return_sequences=1,
                               return_dict_in_generate=True,
                               output_attentions=True,
                               output_scores=True,
                               **gen_kwargs,
            )
            # print('[gen]:', gen)
            # print(tokenizer.batch_decode(gen))
            generated_answers_encoded.append(gen['sequences'][0].detach().cpu())
            encoder_attentions_list.append(gen['encoder_attentions'][0].detach().cpu())
        # print(f'Scores: {generated_answers_encoded}')
        text_results = self.tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
        return text_results, encoder_attentions_list, model_inputs

    def prepare_context_for_visualization(self, context):
        examples, relations = [], []
        response, encoder_outputs, model_input = self.generate_based_on_context(context)
        input_commonsense_relations = model_input.get("input_commonsense_relations")
        encoder_outputs = torch.stack(encoder_outputs)
        n_layers, batch_size, n_heads, src, tgt = encoder_outputs.size()
        print(encoder_outputs.size())
        encoder_attentions = encoder_outputs.view(batch_size, n_layers, n_heads, src, tgt)
        for i, ex in enumerate(encoder_attentions):
            d = {}
            indices = model_input['input_ids'][i].detach().cpu()
            all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
            useful_indeces = indices != self.tokenizer.pad_token_id
            all_tokens = np.array(all_tokens)[useful_indeces]
            all_tokens = [tok.replace('Ġ', '') for tok in all_tokens]
            d['words'] = all_tokens
            d['attentions'] = ex.detach().cpu().numpy()
            examples.append(d)
            relations.append(input_commonsense_relations[i])
        print(d['words'])
        return response, examples, relations

"""
if __name__ == '__main__':
    QUESTION = "What is the color of the sky?"

    models = {
        'relations': './trained_models/relation_facebook-bart-large_3e-05_16/checkpoint-197220',
        'default': './trained_models/default_facebook-bart-large_3e-05_16/checkpoint-198700',
    }

    relation_inf = RelationsInference(
        model_path=models['relations'],
        kg_type=KGType.CONCEPTNET,
        model_type=Model_Type.RELATIONS
    )
    relation_answer = relation_inf.answer_question(QUESTION)

    print('Relations Answer:', relation_answer)
"""