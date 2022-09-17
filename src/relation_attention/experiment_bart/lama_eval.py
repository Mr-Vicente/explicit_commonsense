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
from kgs_binding.conceptnet_handler import ConceptNetHandler
from kgs_binding.kg_qa_binding_utils import load_kg_handler
from data.eval_dataset import EvalDataset, EvalRelationsDataset
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class Eval:
    def __init__(self, args,
                 model_name="facebook/bart-large", experiment_type: Model_Type=Model_Type.DEFAULT,
                 data_type: Data_Type=Data_Type.LAMA, knowledge: KGBaseHandler=None):
        self.device = get_device()
        self.original_knowledge = knowledge
        knowledge = knowledge if knowledge is not None else ConceptNetHandler()
        self.knowledge = knowledge
        self.tokenizer, self.model = self.load_tokenizer_and_model(experiment_type, model_name)
        self.args = args
        dataset = self.load_mask_data(experiment_type, data_type)
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

    def load_mask_data(self, experiment_type:Model_Type, data_type:Data_Type):
        data = read_jsonl_file_2_dict(f'{data_type.value}.jsonl', store_dir='./eval/data')
        if experiment_type == Model_Type.RELATIONS or experiment_type == Model_Type.MASK:
            relation_mapper_builder = RelationsMapperBuilder(knowledge=self.knowledge)
            return EvalRelationsDataset(data, relation_mapper_builder, data_type, self.tokenizer, self.device)
        dataset = EvalDataset(data, data_type, self.tokenizer, self.device)
        return dataset

    def status_print(self, input_ids, preds_txt, label_txt):
        input_txt = self.tokenizer.decode(input_ids, skip_special_tokens=False).strip()
        print()
        print('input_txt:', input_txt.replace('<pad>',''))
        print('preds_txt:', preds_txt)
        print('label_txt:', label_txt)
        print('-----------')


    def batch_eval(self, logits_s, input_ids_s, labels):
        batch_score = defaultdict(float)
        for logits, input_ids, label in zip(logits_s, input_ids_s, labels):
            masked_index = (input_ids.detach().cpu() == self.tokenizer.mask_token_id).nonzero().item()
            probs = logits[masked_index].softmax(dim=0)
            text_label = self.tokenizer.decode(label, skip_special_tokens=True).strip()
            values, predictions = probs.topk(5)
            pred = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            #self.status_print(input_ids, pred, text_label)
            #
            score = self.eval_current(pred, text_label)
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
            words = wordnet.synset(f'{prediction}.n.01')
            words = words.lemma_names()
            cleaned_words = [s.replace('_', ' ') for s in words]
            norm_hard_truth = self.knowledge.normalize_noun(hard_truth)
            for s in cleaned_words:
                norm_s = self.knowledge.normalize_noun(s)
                score = 1 if norm_hard_truth == norm_s else 0
                if score == 1:
                    break
        except Exception as _e:
            score = self.exact_match(prediction, hard_truth)
        return score

    def top_n_score(self, n, preds, label):
        score = 0
        for pred in preds[:n]:
            score = self.calc_relaxed_score(pred, label)
            if score == 1:
                return score
        return score

    def eval_current(self, preds, label):
        #top1_score = 1 if preds[0]==label else 0
        top1_score = self.calc_relaxed_score(preds[0], label)
        score = {'top1_score': top1_score}
        score['top3_score'] = self.top_n_score(3, preds, label)
        score['top5_score'] = self.top_n_score(5, preds, label)
        #if exact_match_score == 0: #and self.original_knowledge is not None:
        #    near_score = 1 if self.knowledge.exists_relation_between(pred, label) else 0
        #    score['near_relation'] = near_score
        #else:
        #    score['near_relation'] = 0
        return score

    def eval(self, dataloader=None):
        if dataloader is None:
            dataloader = self.dataloader
        overall_score = defaultdict(float)
        print('len(dataloader):', len(dataloader))
        print('len(self.dataset):', len(self.dataset))
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader)):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                if "input_commonsense_relations" in batch:
                    input_commonsense_relations = batch["input_commonsense_relations"].to(self.device)
                    logits = self.model(input_ids=input_ids,
                                        input_commonsense_relations=input_commonsense_relations).logits
                else:
                    logits = self.model(input_ids=input_ids).logits
                #logits = self.model(**batch).logits
                score = self.batch_eval(logits, input_ids, labels)
                for metric, s in score.items():
                    overall_score[metric] += s
                wandb.log(score)
            for metric, s in overall_score.items():
                overall_score[metric] = (overall_score[metric] / len(self.dataset) * 100)
            return overall_score

if __name__ == '__main__':
    wandb.init(project="eval", entity="mr-vicente")
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--batch_size", type=int, default=32, help="size of batch")
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
    wandb.log(_score)
    """
    try:
        model_name_str = model_name.split('/')[-2]
    except Exception as _:
        model_name_str = 'facebook/bart-large'
    write_dict_2_json_file(write_score, f'{model_name_str}_eval.json', store_dir=f'eval/results/{dataset_type.value}')
    wandb.finish()
    """

