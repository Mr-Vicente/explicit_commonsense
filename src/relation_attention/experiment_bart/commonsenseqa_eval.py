#############################
#   Imports
#############################

# Python modules
import argparse
from collections import defaultdict
from copy import deepcopy

# Remote modules
import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    default_data_collator,
)
from custom_bart import BartCustomForConditionalGeneration, BartCustomConfig
from custom_tokenizer import BartCustomTokenizerFast
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

from nltk.corpus import wordnet

from tqdm import tqdm

import wandb

# Local modules
from utils import (
    read_jsonl_file_2_dict,
    write_dict_2_json_file,
    Data_Type,
    KGType,
    get_device,
    Model_Type
)
from kgs_binding.kg_base_wrapper import KGBaseHandler, NoKnowledge
from kgs_binding.kg_qa_binding_utils import load_kg_handler
from data.eval_dataset import EvalDataset, EvalRelationsDataset
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder
from data.data_preparation import load_data

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class Eval:
    def __init__(self, args,
                 model_name="facebook/bart-large", experiment_type: Model_Type=Model_Type.DEFAULT,
                 data_type: Data_Type=Data_Type.COMMONSENSE_QA, knowledge: KGBaseHandler=None):
        self.device = get_device()
        self.original_knowledge = knowledge
        knowledge = knowledge if knowledge is not None else NoKnowledge()
        self.knowledge = knowledge
        self.tokenizer, self.model = self.load_tokenizer_and_model(experiment_type, model_name)
        self.args = args
        dataset = self.load_data(experiment_type, data_type)
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        self.dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=default_data_collator
        )
        self.dataset = dataset

    def load_tokenizer_and_model(self, experiment_type: Model_Type, model_name='facebook/bart-large'):
        if experiment_type == Model_Type.MASK or experiment_type == Model_Type.RELATIONS:
            config = BartCustomConfig.from_pretrained(model_name)
            tokenizer = BartCustomTokenizerFast.from_pretrained(model_name)
            relation_names = self.knowledge.get_relation_types()
            tokenizer.set_known_relation_names(relation_names)
            there_is_difference_between_relations = experiment_type==Model_Type.RELATIONS
            tokenizer.set_operation_mode(there_is_difference_between_relations=there_is_difference_between_relations)
            model = BartCustomForConditionalGeneration.from_pretrained(model_name, config=config)
        else:
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def load_data(self, experiment_type:Model_Type, data_type:Data_Type):
        data = read_jsonl_file_2_dict('commonsense_qa.jsonl', store_dir='eval/data')[:100]
        if experiment_type == Model_Type.RELATIONS or experiment_type == Model_Type.MASK:
            relation_mapper_builder = RelationsMapperBuilder(knowledge=self.knowledge)
            return EvalRelationsDataset(data, relation_mapper_builder, data_type, self.tokenizer, self.device)
        dataset = EvalDataset(data, data_type, self.tokenizer, self.device)
        return dataset

    def batch_eval(self, answers, true_answers):
        batch_score = defaultdict(float)
        for answer, true_answer in zip(answers, true_answers):
            score = self.eval_current(answer, true_answer)
            for metric, s in score.items():
                batch_score[metric] += s
        return batch_score

    def exact_match(self, prediction, hard_truth):
        norm_prediction = self.knowledge.normalize_noun(prediction)
        norm_hard_truth = self.knowledge.normalize_noun(hard_truth)
        score = 1 if norm_prediction == norm_hard_truth else 0
        return score

    def calc_relaxed_score(self, prediction, hard_truth):
        score = 0
        try:
            words = wordnet.synset(f'{hard_truth}.n.01')
            words = words.lemma_names()
            cleaned_words = [s.replace('_', ' ') for s in words]
            cleaned_words = list(set(cleaned_words + [hard_truth]))
            for s in cleaned_words:
                norm_s = self.knowledge.normalize_noun(s)
                score = 1 if norm_s in prediction else 0
                if score == 1:
                    break
        except Exception as _e:
            score = 1 if hard_truth in prediction else 0
        return score

    def eval_current(self, answer, true_answer):
        exact_match_score = self.exact_match(answer, true_answer)
        score = {'exact': exact_match_score}
        score['relaxed'] = self.calc_relaxed_score(answer, true_answer)
        return score

    def eval(self, dataloader=None):
        if dataloader is None:
            dataloader = self.dataloader
        overall_score = defaultdict(float)
        print('len(self.dataset):', len(self.dataset))
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader)):
                gen_kwargs = {}
                true_answers = batch['labels']
                if "input_commonsense_relations" in batch:
                    print(batch['input_commonsense_relations'].sum())
                    gen_kwargs["relation_inputs"] = batch.get("input_commonsense_relations").to(self.device)
                generated_answers_encoded = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    min_length=1,
                    max_length=128,
                    do_sample=True,
                    early_stopping=True,
                    num_beams=8,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    no_repeat_ngram_size=2,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **gen_kwargs,
                )
                # print(f'Scores: {generated_answers_encoded}')
                answers = self.tokenizer.batch_decode(generated_answers_encoded['sequences'], skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)
                true_answers_str = self.tokenizer.batch_decode(true_answers, skip_special_tokens=True)
                if idx % 2 == 0:
                    context_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    print('Context - answers - real answers:')
                    print(f'{context_str[0]}|{answers[0]}|{true_answers_str[0]}')
                score = self.batch_eval(answers, true_answers_str)
                for metric, s in score.items():
                    overall_score[metric] += s
                #wandb.log(score)
            for metric, s in overall_score.items():
                overall_score[metric] = (overall_score[metric] / len(self.dataset) * 100)
            return overall_score

if __name__ == '__main__':
    #wandb.init(project="eval", entity="mr-vicente")
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--batch_size", type=int, default=16, help="size of batch")
    parser.add_argument("--model_path", type=str, default="facebook/bart-large", help="path for the model")
    parser.add_argument("--experiment_type", type=str, choices=["default", "relations", "mask"], default="default",
                        help="model type")
    parser.add_argument("--knowledge", type=str, choices=["conceptnet", "swow", "cskg"], default=None,
                        help="uses knowledge")
    parser.add_argument("--dataset", type=str, choices=["lama", "commonsense_qa"], default="lama",
                        help="eval dataset")

    args = parser.parse_args()

    dataset_type = Data_Type(args.dataset)
    EXPERIMENT_TYPE: Model_Type = Model_Type(args.experiment_type)
    knowledge_handler: KGBaseHandler = load_kg_handler(KGType(args.knowledge)) if args.knowledge else None
    model_name = args.model_path

    e = Eval(args,
             model_name=model_name, experiment_type=EXPERIMENT_TYPE,
             data_type=dataset_type, knowledge=knowledge_handler)
    _score = e.eval()
    write_score = dict(deepcopy(_score))
    print(write_score)
    #wandb.log(_score)
    try:
        model_name_str = model_name.split('/')[-2]
    except Exception as _:
        model_name_str = 'facebook/bart-large'
    write_dict_2_json_file(write_score, f'{model_name_str}_eval.json', store_dir=f'eval/results/{dataset_type.value}')
    #wandb.finish()

