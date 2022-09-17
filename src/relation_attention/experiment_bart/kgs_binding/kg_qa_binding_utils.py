#############################
#   Imports
#############################

# Python modules
from typing import List, Tuple
from enum import Enum

# Remote modules

# Local modules
from .kg_base_wrapper import KGBaseHandler
from .cskg_handler import CSKGHandler
from .swow_handler import SwowHandler
from .conceptnet_handler import ConceptNetHandler
from utils import read_json_file_2_dict, Data_Type

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class KGType(Enum):
    SWOW = 'swow'
    CSKG = 'cskg'
    CONCEPTNET = 'conceptnet'

def load_kg_handler(kg_type: KGType):
    if kg_type.value == KGType.SWOW.value:
        return SwowHandler()
    elif kg_type.value == KGType.CONCEPTNET.value:
        return ConceptNetHandler()
    elif kg_type.value == KGType.CSKG.value:
        return CSKGHandler()
    else:
        raise NotImplementedError()

def _load_data_paths_metadata():
    try:
        data = read_json_file_2_dict('data_config.json', store_dir='run_config')
    except:
        data = None
    return data

def from_relations_path_2_relations(dataset_types: List[Data_Type], metadata):
    relations = []
    print('metadata:', metadata)
    for dataset_type in dataset_types:
        qa_meta_data = metadata[dataset_type.value]
        filename_path, dir_data = qa_meta_data['local']
        print(filename_path, dir)
        data = read_json_file_2_dict(filename_path, dir_data)
        relations.extend(data)
    return relations

def KGHandler_to_str(kg_handler: KGBaseHandler) -> str:
    if isinstance(kg_handler, SwowHandler):
        return 'swow'
    elif isinstance(kg_handler, ConceptNetHandler):
        return 'conceptnet'
    elif isinstance(kg_handler, CSKGHandler):
        return 'cskg'
    else:
        raise NotImplementedError()

def get_kg_qa_data_metadata(kg_handler: KGBaseHandler) -> Tuple[str, str]:
    kg_qa_data_path = _load_data_paths_metadata()
    if isinstance(kg_handler, SwowHandler):
        swow = kg_qa_data_path["swow"]
        return swow
    elif isinstance(kg_handler, ConceptNetHandler):
        conceptnet = kg_qa_data_path["conceptnet"]
        return conceptnet
    elif isinstance(kg_handler, CSKGHandler):
        cskg = kg_qa_data_path["cskg"]
        return cskg
    else:
        raise NotImplementedError()