#############################
#   Imports
#############################

# Python modules
import argparse
from typing import List, Dict
from distutils.util import strtobool

# Remote modules
import torch
from transformers import BartTokenizer

# Local modules
from kgs_binding import RelationsMapperBuilder
from utils import Data_Type, Model_Type, KGType, get_device, create_directory
from custom_tokenizer.bart_custom_tokenizer_fast import BartCustomTokenizerFast
from kgs_binding.kg_qa_binding_utils import load_kg_handler
from kgs_binding.kg_base_wrapper import KGBaseHandler
from kgs_binding.kg_qa_binding_utils import get_kg_qa_data_metadata, from_relations_path_2_relations
from data.RelationsDatasets import RelationsDataset, MaskRelationsDataset
from data.data_preparation import load_and_preprocess_data
from data.DatasetGeneral import DatasetGeneral


from data.datasets_model_handling import (
    DatasetParsingUtils,
    DatasetRelationsParsingUtils
)

#############################
#   Constants
#############################

MODEL_NAME = "facebook/bart-large"

#############################
#   Helper functions
#############################

def handler_program_arguments():
    print('-----Argument parsing------')
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs='+', type=str, default=["eli5", "stackexchange_qa"],
                        help="datasets to train")
    parser.add_argument("--experiment_type", type=str, choices=["default", "relations", "mask"], default="mask",
                        help="model type")
    parser.add_argument("--knowledge", type=str, choices=["conceptnet", "swow", "cskg"], default="swow",
                        help="uses knowledge")
    parser.add_argument("--max_length", type=int, default=128,
                        help="max length of sequence")
    parser.add_argument("--use_extra_relations", type=lambda x: bool(strtobool(x)), default="true",
                        help="map rest of input space with kg relations")
    parser.add_argument("--use_context", type=lambda x: bool(strtobool(x)), default="true",
                        help="use context of dataset")
    parser.add_argument("--relations_data_path", type=str, default=None,
                        help="use pre generated relations data")
    parser.add_argument("--pre_training_model", type=str, default="facebook/bart-large",
                        help="use pre generated relations data")
    parser.add_argument("--should_tokenize_default", type=lambda x: bool(strtobool(x)), default="false",
                        help="use context of dataset")

    args = parser.parse_args()
    return args

def get_tokenized_data(model_type:Model_Type,
                       datasets_parsing_utils:DatasetRelationsParsingUtils,
                       tokenizer,
                       device,
                       max_length,
                       relations_data):
    if model_type == Model_Type.RELATIONS:
        dataset = RelationsDataset(
            datasets_parsing_utils=datasets_parsing_utils,
            tokenizer=tokenizer, device=device,
            relations_data=relations_data,
            max_length=max_length
        )
    elif model_type == Model_Type.MASK:
        dataset = MaskRelationsDataset(
            datasets_parsing_utils=datasets_parsing_utils,
            tokenizer=tokenizer, device=device,
            relations_data=relations_data,
            max_length=max_length
        )
    else:
        raise NotImplementedError()
    return dataset.tokenized_data

def store_tok_data(data, data_path):
    dirs = data_path.split('/')[:-1]
    dirs_str = '/'.join(dirs)
    create_directory(dirs_str)
    path = f'{data_path}.pkl'
    #with open(path, 'wb') as f:
    torch.save(data, path)


#############################
#   Main
#############################
if __name__ == '__main__':
    args = handler_program_arguments()

    # get device
    device = get_device()

    # pre_processing_args (turn strs into enums)
    print(args)
    DATASETS_CONSIDERED: List[Data_Type] = [Data_Type(dataset_type) for dataset_type in args.datasets]
    EXPERIMENT_TYPE: Model_Type = Model_Type(args.experiment_type)
    KG_Handler: KGBaseHandler = load_kg_handler(KGType(args.knowledge)) if args.knowledge else None

    tokenizer = BartCustomTokenizerFast.from_pretrained(args.pre_training_model)
    simple_tokenizer = BartTokenizer.from_pretrained(args.pre_training_model)
    relation_names = KG_Handler.get_relation_types()
    tokenizer.set_known_relation_names(relation_names)
    diff = EXPERIMENT_TYPE.there_is_difference_between_relations()
    tokenizer.set_operation_mode(there_is_difference_between_relations=diff)

    relations_data = None
    if args.relations_data_path:
        relations_metadata = get_kg_qa_data_metadata(KG_Handler)
        relations_data = from_relations_path_2_relations(DATASETS_CONSIDERED, metadata=relations_metadata)

    # data loading
    training_data, validation_data, testing_data = load_and_preprocess_data(DATASETS_CONSIDERED)

    # Define storing data info
    store_dir = f'/home/fm.vicente/data/tok_data'
    store_dir_simple = f'/home/fm.vicente/data/tok_data_simple'
    KG_str = args.knowledge
    datasets_str = Data_Type.data_types_to_str(DATASETS_CONSIDERED)
    data_type = DATASETS_CONSIDERED[0]
    max_length = args.max_length
    exp_str = args.experiment_type
    use_context = args.use_context
    use_extra_relations = args.use_extra_relations
    use_context_str = '_wContext' if args.use_context else ''
    use_extra_rels_str = '_wExtraRels' if args.use_extra_relations else ''

    relations_mapper = RelationsMapperBuilder(knowledge=KG_Handler,
                                              datatype=data_type)

    # data handling
    print("====Data Tokenizing====")
    print("Training set")
    datasets_parsing_utils = DatasetRelationsParsingUtils(
        dataset_type=data_type,
        data=training_data,
        relations_mapper=relations_mapper,
        use_extra_relations=use_extra_relations,
        use_context=use_context
    )
    training_tok_data = get_tokenized_data(EXPERIMENT_TYPE, datasets_parsing_utils, tokenizer, device, max_length, relations_data)
    print("Storing data")
    training_data_path = f"{store_dir}/{datasets_str}/{exp_str}_training_{KG_str}{use_context_str}{use_extra_rels_str}"
    store_tok_data(training_tok_data, training_data_path)
    del training_tok_data

    if args.should_tokenize_default:
        training_data_simple_path = f"{store_dir_simple}/{datasets_str}/training{use_context_str}"
        training_dataset_parsing = DatasetParsingUtils(data_type, training_data, use_context=use_context)
        simple_tok_training_data = DatasetGeneral(training_dataset_parsing,tokenizer=simple_tokenizer, device=device, max_length=max_length)
        store_tok_data(simple_tok_training_data, training_data_simple_path)

    print("Validation set")
    datasets_parsing_utils = DatasetRelationsParsingUtils(
        dataset_type=data_type,
        data=validation_data,
        relations_mapper=relations_mapper,
        use_extra_relations=use_extra_relations,
        use_context=use_context
    )
    validation_tok_data = get_tokenized_data(EXPERIMENT_TYPE, datasets_parsing_utils, tokenizer, device, max_length, relations_data)
    print("Storing data")
    validation_data_path = f"{store_dir}/{datasets_str}/{exp_str}_validation_{KG_str}{use_context_str}{use_extra_rels_str}"
    store_tok_data(validation_tok_data, validation_data_path)
    del validation_tok_data

    if args.should_tokenize_default:
        training_data_simple_path = f"{store_dir_simple}/{datasets_str}/validation{use_context_str}"
        training_dataset_parsing = DatasetParsingUtils(data_type, validation_data, use_context=use_context)
        simple_tok_training_data = DatasetGeneral(training_dataset_parsing,tokenizer=simple_tokenizer, device=device, max_length=max_length)
        store_tok_data(simple_tok_training_data, training_data_simple_path)

    print("Testing set")
    datasets_parsing_utils = DatasetRelationsParsingUtils(
        dataset_type=data_type,
        data=testing_data,
        relations_mapper=relations_mapper,
        use_extra_relations=use_extra_relations,
        use_context=use_context
    )
    testing_tok_data = get_tokenized_data(EXPERIMENT_TYPE, datasets_parsing_utils, tokenizer, device, max_length, relations_data)
    print("Storing data")
    testing_data_path = f"{store_dir}/{datasets_str}/{exp_str}_testing_{KG_str}{use_context_str}{use_extra_rels_str}"
    store_tok_data(testing_tok_data, testing_data_path)
    del testing_tok_data

    if args.should_tokenize_default:
        training_data_simple_path = f"{store_dir_simple}/{datasets_str}/testing{use_context_str}"
        training_dataset_parsing = DatasetParsingUtils(data_type, training_data, use_context=use_context)
        simple_tok_training_data = DatasetGeneral(training_dataset_parsing,tokenizer=simple_tokenizer, device=device, max_length=max_length)
        store_tok_data(simple_tok_training_data, training_data_simple_path)

    print("== All done :)) ==")
