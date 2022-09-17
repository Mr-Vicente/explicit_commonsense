#############################
#   Imports
#############################

# Python modules

# Remote modules

# Local modules
from src.utils import read_json_file_2_dict, write_dict_2_json_file

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

data_kg = 'swow'

if __name__ == '__main__':
    filenames_to_join = ['train_eli5_swow_relation_data.json',
                         'validation_eli5_swow_relation_data.json',
                         'test_eli5_swow_relation_data.json']
    store_content = []
    for f in filenames_to_join:
        content = read_json_file_2_dict(f, store_dir=data_kg)
        store_content.extend(content)
    store_filename = '_'.join(filenames_to_join[0].split('_')[1:])
    write_dict_2_json_file(store_content, store_filename, store_dir=data_kg)
