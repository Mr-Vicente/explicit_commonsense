
#############################
#   Imports
#############################

# Python modules
import sys
from typing import List
import argparse
# Remote modules

# Local modules
from qa import Bart_QA
from utils import Data_Type, Model_Type, Head_Mask, KGType, read_json_file_2_dict
from kgs_binding.kg_base_wrapper import KGBaseHandler
from kgs_binding.cskg_handler import CSKGHandler
from kgs_binding.swow_handler import SwowHandler
from kgs_binding.conceptnet_handler import ConceptNetHandler
from kgs_binding.kg_qa_binding_utils import load_kg_handler
#from run_args import args

# performance and logging

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
    parser.add_argument("--experiment_type", type=str, choices=["default", "relations", "mask"], default="relations", help="model type")
    parser.add_argument("--knowledge", type=str, choices=["conceptnet", "swow", "cskg"], default="conceptnet", help="uses knowledge")
    parser.add_argument("--head_mask_type", type=str, choices=["all", "random", "specific", "none"], default="none", help="what type of mask")
    parser.add_argument("--specific_heads", type=list, nargs='+', default=None, help="mask specific heads")

    args = parser.parse_args()

    # pre_processing_args (turn strs into enums)
    print(args.datasets)
    DATASETS_CONSIDERED: List[Data_Type] = [Data_Type(dataset_type) for dataset_type in args.datasets]
    EXPERIMENT_TYPE: Model_Type = Model_Type(args.experiment_type)
    KG_Handler: KGBaseHandler = load_kg_handler(KGType(args.knowledge)) if args.knowledge else None
    HEAD_MASK_TYPE = Head_Mask(args.head_mask_type) if args.head_mask_type else None
    SPECIFIC_HEADS = args.specific_heads

    print('-----Starting BART------')
    model = Bart_QA(model_name="facebook/bart-large", experiment_type=EXPERIMENT_TYPE, kg_handler=KG_Handler,
                    heads_mask_type=HEAD_MASK_TYPE,
                    specific_heads=SPECIFIC_HEADS,
                    run_config=None
    )
    print('Choosing datasets')
    train_dataset, validation_dataset, test_dataset = model.load_data(
        DATASETS_CONSIDERED,
        EXPERIMENT_TYPE
    )
    model.train(train_dataset, validation_dataset, test_dataset, report=True)
    print('----- All done :)))) ------')
