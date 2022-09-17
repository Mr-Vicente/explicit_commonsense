
#############################
#   Imports
#############################

# Python modules
import sys
from typing import List
import argparse
from distutils.util import strtobool

# Remote modules

# Local modules
#from qa import Bart_QA
from qa_model import Bart_QA
from utils import (
    Data_Type,
    Model_Type,
    Head_Mask,
    KGType,
    read_json_file_2_dict,
    LossType,
    ScoringType,
    CURRENT_PRETRAINING_NAME
)
from kgs_binding.kg_base_wrapper import KGBaseHandler
from kgs_binding.cskg_handler import CSKGHandler
from kgs_binding.swow_handler import SwowHandler
from kgs_binding.conceptnet_handler import ConceptNetHandler
from kgs_binding.kg_qa_binding_utils import load_kg_handler
#from run_args import args

# performance and logging
import wandb
#from run_config.overide_pynvml import setup

#############################
#   Constants
#############################

#################### Remote testing ####################
gettrace = getattr(sys, 'gettrace', None)
if gettrace is None:
    print('No sys.gettrace')
elif gettrace():
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=9000, stdoutToServer=True, stderrToServer=True)
#################### Remote testing ####################

if __name__ == '__main__':
    print('-----Argument parsing------')
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs='+', type=str, default=["eli5", "stackexchange_qa"], help="datasets to train")
    parser.add_argument("--experiment_type", type=str, choices=["default", "relations", "mask"], default="default", help="model type")
    parser.add_argument("--knowledge", type=str, choices=["conceptnet", "swow", "cskg"], default=None, help="uses knowledge")
    parser.add_argument("--head_mask_type", type=str, choices=["all", "random", "specific", "none"], default=None, help="what type of mask")
    parser.add_argument("--specific_heads", type=list, nargs='+', default=None, help="mask specific heads")
    parser.add_argument("--loss_type", type=str, choices=["default", "cp-rp-def", "cp-def"], default="default", help="Define which loss function variation to use")
    parser.add_argument("--scoring_type", type=str, choices=["default", "max-prob", "interpol", "multiple_choice", "constraint"], default="default",
                        help="Define which kg decoding strategy to use")
    parser.add_argument("--use_context", type=lambda x: bool(strtobool(x)), default="no", help="use context of dataset")
    parser.add_argument("--use_extra_relations", type=lambda x: bool(strtobool(x)), default="no",
                        help="use extra relations (this not actually changing anything, just here for runname, to change go to pre_tokenization file)")
    parser.add_argument("--use_dynamic_heads", type=lambda x: bool(strtobool(x)), default="no",
                        help="change heads overtime to make use of irrelevant heads to commonsense")
    parser.add_argument("--learn_pos_embed_encoder", type=lambda x: bool(strtobool(x)), default="yes", help="use context of dataset")
    parser.add_argument("--pre_model", type=str, default="facebook/bart-large", choices=["facebook/bart-large", "facebook/bart-base", "patrickvonplaten/bart-large-fp32"], help="Pre-trained model")

    parser.add_argument("--epochs", type=int, default=10, help="N epochs for training")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--max_length", type=int, default=32, help="max generation sequence length")
    parser.add_argument("--lr", type=float, default=None, help="define learning rate")

    args = parser.parse_args()

    # pre_processing_args (turn strs into enums)
    print(args.datasets)
    DATASETS_CONSIDERED: List[Data_Type] = [Data_Type(dataset_type) for dataset_type in args.datasets]
    EXPERIMENT_TYPE: Model_Type = Model_Type(args.experiment_type)
    KG_Handler: KGBaseHandler = load_kg_handler(KGType(args.knowledge)) if args.knowledge else None
    HEAD_MASK_TYPE = Head_Mask(args.head_mask_type) if args.head_mask_type else None
    SPECIFIC_HEADS = args.specific_heads
    loss_type = LossType(args.loss_type)
    kg_decoding_scoring_type = ScoringType(args.scoring_type)
    pre_trained_model = args.pre_model.replace('/','-') if args.pre_model else ''

    # run info
    runs_info = read_json_file_2_dict('runs_info.json', store_dir='runs')
    HEAD_MASK_TYPE_STR = args.head_mask_type if args.head_mask_type else 'none'
    KNOWLEDGE_TYPE_STR = args.knowledge if args.knowledge else 'none'
    datasets_str = '-'.join(args.datasets)
    exp_type = EXPERIMENT_TYPE.value
    learn_pos_embed_encoder_str = '_wLearnEmb' if args.learn_pos_embed_encoder else ''
    use_context_str = '_wContext' if args.use_context else ''
    use_extra_rels_str = '_wExtraRels' if args.use_extra_relations else ''
    use_dynamic_heads_str = '_wDynamicHeads' if args.use_dynamic_heads else ''
    loss_type_str = f'L-{args.loss_type}' if args.loss_type else 'L-default'
    kg_decoding_scoring_type_str = f'DS-{args.scoring_type}' if args.scoring_type else 'DS-default'
    run_uri = f'{pre_trained_model}_{exp_type}_{datasets_str}_{KNOWLEDGE_TYPE_STR}_{HEAD_MASK_TYPE_STR}_{loss_type_str}_{kg_decoding_scoring_type_str}{use_context_str}{use_extra_rels_str}{learn_pos_embed_encoder_str}'
    idx = runs_info.get(f'{run_uri}', 0)
    run_name = f'{run_uri}_{idx}'
    run_config = {
        'run_name': run_uri,
        'datasets': datasets_str,
        'kg': KNOWLEDGE_TYPE_STR,
        'mask': HEAD_MASK_TYPE_STR,
        'num_train_epochs': args.epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'generation_max_length': args.max_length
    }
    print('-----Weights and biases init------')
    wandb.init(project=f"bart_{exp_type}",
               entity="mr-vicente",
               name=run_name)
    print(f'WandB initiated: {run_name}')

    # cuda setup
    #setup()

    print('-----Starting BART------')
    model = Bart_QA(model_name=args.pre_model, experiment_type=EXPERIMENT_TYPE, kg_handler=KG_Handler,
                    heads_mask_type=HEAD_MASK_TYPE,
                    specific_heads=SPECIFIC_HEADS,
                    run_config=run_config,
                    args=args
    )
    print('Choosing datasets')
    train_dataset, validation_dataset, test_dataset = model.load_data(
        dataset_types=DATASETS_CONSIDERED,
        experiment_type=EXPERIMENT_TYPE,
        use_context_str=use_context_str,
        use_extra_rels_str=use_extra_rels_str
    )
    model.train(train_dataset, validation_dataset, test_dataset, loss_type, kg_decoding_scoring_type, report=True)
    wandb.finish()
    print('----- All done :)))) ------')
