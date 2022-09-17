#############################
#   Imports
#############################

# Python modules
from enum import Enum
import argparse

# Remote modules

# Local modules
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder
from kgs_binding.swow_handler import SwowHandler
from kgs_binding.cskg_handler import CSKGHandler
from kgs_binding.conceptnet_handler import ConceptNetHandler
from utils import Data_Type,  write_dict_2_json_file, KGType

#############################
#   Constants
#############################

def select_kg(kg_type: KGType = KGType.SWOW):
    if kg_type.value == KGType.SWOW.value:
        return SwowHandler()
    elif kg_type.value == KGType.CSKG.value:
        return CSKGHandler()
    elif kg_type.value == KGType.CONCEPTNET.value:
        return ConceptNetHandler("/Users/mrvicente/Documents/Education/Thesis/code/conceptnet.db")
    else:
        raise NotImplementedError()

data_dict = {
    'eli5': {
        'local': ('validation_eli5.json', '/Users/mrvicente/Documents/Education/Thesis/code/f_papers/explicit_commonsense/src/data'),
        'remote': ('validation_eli5.json', '/home/fm.vicente/explicit_commonsense/src/data')
    },
    'stackexchange_qa': {
        'local': ('stackexchange_final.json', '/Users/mrvicente/Documents/Education/Thesis/code/f_papers/explicit_commonsense/src/data'),
        'remote': ('stackexchange_final.json', '/home/fm.vicente/explicit_commonsense/src/data')
    },
    'commongen_qa': {
        'local': ('commongen_qa_final.json', '/Users/mrvicente/Documents/Education/Thesis/code/f_papers/explicit_commonsense/src/relation_attention'),
        'remote': ('commongen_qa_final.json', '/home/fm.vicente/explicit_commonsense/src/data')
    },
    'commonsense_qa': {
        'local': ('commonsense_qa_final.json', '/Users/mrvicente/Documents/Education/Thesis/code/f_papers/explicit_commonsense/src/data'),
        'remote': ('commonsense_qa_final.json', '/home/fm.vicente/explicit_commonsense/src/data')
    },
    'commongen': {
        'local': ('commongen.json','/Users/mrvicente/Documents/Education/Thesis/code/f_papers/explicit_commonsense/src/data'),
        'remote': ('commongen.json','/home/fm.vicente/explicit_commonsense/src/data')
    },
}

if __name__ == '__main__':

    print('-----Argument parsing------')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["commongen", "eli5", "commonsense_qa", "stackexchange_qa"], default="commongen", help="dataset")
    parser.add_argument("--knowledge", type=str, choices=["conceptnet", "swow", "cskg"], default="conceptnet", help="knowledge")
    args = parser.parse_args()

    SELECTED_KG = KGType(args.knowledge) if args.knowledge else None
    DATA_TYPE = Data_Type(args.dataset) if args.dataset else None
    kg_handler = select_kg(kg_type=SELECTED_KG)
    FILE_NAME, FILE_DIR = data_dict[DATA_TYPE.value]['remote']
    relations_mapper = RelationsMapperBuilder(knowledge=kg_handler,
                                              filename=FILE_NAME,
                                              file_dir=FILE_DIR,
                                              datatype=DATA_TYPE)
    relations_mappings = relations_mapper.get_relations_mapping_complex(clear_common_wds=True)
    type_file = FILE_NAME.split('_')[0]
    write_dict_2_json_file(relations_mappings,
                           f'kgs_binding/model_kgs/{type_file}_{DATA_TYPE.value}_{SELECTED_KG.value}_relation_data.json')
    print('relations_mappings:', relations_mappings)