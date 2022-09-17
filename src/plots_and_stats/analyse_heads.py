#############################
#   Imports
#############################

# Python modules
import argparse

# Remote modules
import torch

# Local modules
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    default_data_collator
)

from attention_viz import AttentionVisualizer
from DatasetGeneral import DatasetGeneral
from QA_Datasets import RelationsDataset
from data_preparation import load_and_preprocess_data
from utils import Data_Type

#############################
#   Constants
#############################

model_name = 'facebook/bart-large'

def load_tokenizer_and_model():
    #tokenizer = BartTokenizer.from_pretrained(model_name)
    tokenizer = BartCustomTokenizerFast.from_pretrained(model_name)
    k = ConceptNetHandler("/Users/mrvicente/Documents/Education/Thesis/code/conceptnet.db")
    relation_names = k.get_relation_types()
    tokenizer.set_known_relation_names(relation_names)
    tokenizer.set_operation_mode(there_is_difference_between_relations=True)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    model.zero_grad()
    return tokenizer, model

if __name__ == '__main__':
    att_viz = AttentionVisualizer()
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--device", type=str, default="cpu", help="device to be used: cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=8, help="size of batch.")
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )

    args = parser.parse_args()

    training_data, validation_data, test_data = load_and_preprocess_data([Data_Type.ELI5, Data_Type.STACK_EXCHANGE])
    #dataset = DatasetGeneral(training_data[:32], tokenizer=tokenizer, device=args.device)
    dataset = RelationsDataset(training_data, tokenizer=tokenizer, device=args.device)
    sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=default_data_collator
    )
    att_viz.compute_heads_importance(args, model, dataloader, commonsense_head_analysis=True)