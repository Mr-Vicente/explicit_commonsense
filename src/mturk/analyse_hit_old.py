#############################
#   Imports
#############################

# Python modules
import json
import pandas as pd
import argparse
# Remote modules

# Local modules

#############################
#   Constants
#############################
def write_dict_2_json_file(json_object, filename, store_dir='.'):
    with open(f'{store_dir}/{filename}', 'w', encoding='utf-8') as file:
        json.dump(json_object, file, ensure_ascii=False, indent=4)
#############################
#   Stuff
#############################
def get_args():
    print('-----Argument parsing------')
    parser = argparse.ArgumentParser()

    parser.add_argument("--absqa_models_paths", nargs='+', type=str, default=None, help="")
    parser.add_argument("--commongen_models_paths", nargs='+', type=str, default=None, help="")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    commongen_models_paths = args.commongen_models_paths
    absqa_models_paths = args.absqa_models_paths
    n_commongen_models = len(commongen_models_paths)
    n_absqa_models = len(absqa_models_paths)

    df = pd.read_csv('amazon_hit_results.csv')

    human_sentences = list(df['human_sentence'])
    write_dict_2_json_file(human_sentences, 'human_sentences', 'results_processed')

    cs_relevant_collumns = [f'commonsense_rating_model_{i}' for i in range(n_commongen_models)]
    descriptive_relevant_collumns = [f'descriptive_rating_model_{i}' for i in range(n_commongen_models)]
    absqa_relevant_collumns = [f'absqa_rating_model_{i}' for i in range(n_absqa_models)]

    cs_data = df[cs_relevant_collumns]
    descriptive_data = df[descriptive_relevant_collumns]
    absqa_data = df[absqa_relevant_collumns]

    cs_data_mean = cs_data.mean(axis=0)
    descriptive_data_mean = descriptive_data.mean(axis=0)
    absqa_data_mean = absqa_data.mean(axis=0)

    print('cs_data_mean:\n', cs_data_mean)
    print("=======\n")

    print('descriptive_data_mean:\n', descriptive_data_mean)
    print("=======\n")

    print('absqa_data_mean:\n', absqa_data_mean)
    print("=======\n")

