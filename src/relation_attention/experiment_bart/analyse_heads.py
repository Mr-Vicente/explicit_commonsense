#############################
#   Imports
#############################

# Python modules
import argparse

# Remote modules
import torch
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    default_data_collator
)
import wandb


# Local modules
from heads_analysis.attention_viz import AttentionVisualizer
from data.DatasetGeneral import DatasetGeneral
from data.relations_datasets_preprocessing import (
    RelationsDataset,
    MaskRelationsDataset
)#RelationsDatasets import RelationsDataset
from data.eval_dataset import EvalDataset, EvalRelationsDataset
from data.data_preparation import load_and_preprocess_data, load_csv_data, load_tok_data
from kgs_binding import RelationsMapperBuilder
from utils import Data_Type, KGType, get_device, read_jsonl_file_2_dict

from kgs_binding.kg_qa_binding_utils import get_kg_qa_data_metadata, from_relations_path_2_relations, load_kg_handler

from custom_tokenizer.bart_custom_tokenizer_fast import BartCustomTokenizerFast
from custom_bart.bart_for_conditional_generation import BartCustomForConditionalGeneration
from custom_bart import BartCustomConfig

#############################
#   Constants
#############################

model_name = 'facebook/bart-large'
#model_name = './trained_models/default_none_none_2_facebook-bart-large_3e-05_16/checkpoint-198700'

def load_tokenizer_and_model(knowledge_handler, device):
    #tokenizer = BartTokenizer.from_pretrained(model_name)
    tokenizer = BartCustomTokenizerFast.from_pretrained(model_name)
    relation_names = knowledge_handler.get_relation_types()
    tokenizer.set_known_relation_names(relation_names)
    tokenizer.set_operation_mode(there_is_difference_between_relations=False)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    #config = BartCustomConfig()
    #model = BartCustomForConditionalGeneration(model_name, config=config)
    #model.eval()
    #model.zero_grad()
    model.to(device)
    return tokenizer, model

if __name__ == '__main__':
    wandb.init(project="heads", entity="mr-vicente")

    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--device", type=str, default="cuda", help="device to be used: cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=32, help="size of batch.")
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_false",
        help="Don't normalize all importance scores between 0 and 1",
    )

    args = parser.parse_args()
    device = get_device()
    args.device = device
    att_viz = AttentionVisualizer(device)

    store_dir = f'/home/fm.vicente/data/tok_data'
    #dataset_types = [Data_Type.ELI5, Data_Type.STACK_EXCHANGE]
    training_data = load_tok_data('training', exp_type='relations', datasets='commongen', store_dir=store_dir, kg='conceptnet',
                                  use_context_str='', use_extra_rels_str='')
    train_dataset = RelationsDataset(training_data, device=device)
    knowledge_handler = load_kg_handler(kg_type=KGType.CONCEPTNET)
    tokenizer, model = load_tokenizer_and_model(knowledge_handler, args.device)

    """
    training_data, validation_data, test_data = load_and_preprocess_data(dataset_types, limit=10000)
    relations_metadata = get_kg_qa_data_metadata(knowledge_handler)
    relations_data = from_relations_path_2_relations(dataset_types, metadata=relations_metadata)
    #print('relations_data[:3]:', relations_data[:3])
    dataset = RelationsDataset(training_data, relations_data=relations_data, tokenizer=tokenizer, device=args.device)
    """
    """
    data = read_jsonl_file_2_dict(f'lama.jsonl', store_dir='./eval/data')
    data_type = Data_Type.LAMA
    dataset = EvalDataset(data, data_type, tokenizer, args.device)
    """
    #data = load_csv_data(f'heads_analysis/custom_analysis_data/relations_fix.txt')
    #data_type = Data_Type.CUSTOM
    # Falta o relationship mapper
    #relation_mapper_builder = RelationsMapperBuilder(knowledge=knowledge_handler)
    #dataset = EvalRelationsDataset(data, relation_mapper_builder, data_type, tokenizer, args.device)
    sampler = SequentialSampler(training_data) if args.local_rank == -1 else DistributedSampler(training_data)
    dataloader = DataLoader(
        training_data, sampler=sampler, batch_size=args.batch_size, collate_fn=default_data_collator
    )
    att_viz.compute_heads_importance(args, model, dataloader, commonsense_head_analysis=True)