#############################
#   Imports
#############################

# Python modules
import json
import pandas as pd
import argparse
from ast import literal_eval
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

def print_rating_occurence(collumns, rating_value):
    for collumn in collumns:
        count = (df[collumn]==rating_value).sum()
        print(f'{rating_value}-{collumn}', count)

if __name__ == '__main__':
    args = get_args()
    commongen_models_paths = args.commongen_models_paths
    absqa_models_paths = args.absqa_models_paths
    n_commongen_models = len(commongen_models_paths)
    n_absqa_models = len(absqa_models_paths)

    #df_base = pd.read_csv('results/amazon_hit_results.csv')#Batch_4816114_batch_results-3.csv
    df_base = pd.read_csv('results/Batch_4816114_batch_results-11.csv')
    n=131
    #results = [list(df_base['Answer.taskAnswers'].map(lambda a: literal_eval(a)[0]))[n]]
    df_base = df_base.loc[df_base['AssignmentStatus'] == 'Approved']
    results = list(df_base['Answer.taskAnswers'].map(lambda a: literal_eval(a)[0]))
    print(len(results))
    df = pd.DataFrame(results)
    print(df)

    human_data = {
        'commonsense_model_0': list(df_base['Input.commonsense_model_0']),
        'commonsense_model_1': list(df_base['Input.commonsense_model_1']),
        'commonsense_model_2': list(df_base['Input.commonsense_model_2']),
        'commonsense_model_3': list(df_base['Input.commonsense_model_3']),
        'commonsense_model_4': list(df_base['Input.commonsense_model_4']),
        'concepts_input': list(df_base['Input.step3_concepts']),
        'human_sentences': list(df['human_sentence'])
    }
    #human_sentences = list(df['human_sentence'])
    write_dict_2_json_file(human_data, 'human_sentences.json', 'results_processed')

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


    for rating_value in range(1,6):
        print_rating_occurence(cs_relevant_collumns, rating_value)

    print()
    for rating_value in range(1,6):
        print_rating_occurence(descriptive_relevant_collumns, rating_value)

    print()
    for rating_value in range(1,6):
        print_rating_occurence(absqa_relevant_collumns, rating_value)

    mean_time_in_minutes = df_base['WorkTimeInSeconds'].mean() // 60
    print('\n mean_time_in_minutes:', mean_time_in_minutes)
    #print(human_sentences)
    #print(list(df_base['Input.step3_concepts'])[n])

