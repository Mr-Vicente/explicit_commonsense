
#############################
#   Imports
#############################

# Python modules
import os
import argparse

# Remote modules
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

# Local modules
from utils import read_json_file_2_dict, write_dict_2_json_file

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

def plot_results(title, df, y_label, data_name):
    df.plot.bar(rot=10, title="Test")
    plt.title(f'{title}')
    plt.xlabel('Run types')
    plt.ylabel(y_label)
    plt.rc('axes', axisbelow=True)
    plt.grid(True)
    #plt.savefig(f'eval/plots/{title}.png')
    plt.savefig(f'eval/plots/{data_name}/{data_name}_eval.png')
    plt.close()

def load_results_data(data_dir='eval/results'):
    run_names, results = [], []
    for f_name in os.listdir(data_dir):
        result = read_json_file_2_dict(f_name, data_dir)
        results.append(result)
        run_name = f_name.replace('.json', '')
        try:
            run_name = '_'.join(run_name.split('_')[:-5])
        except Exception as _:
            run_name = 'facebook/bart-large'
        run_names.append(run_name)
    return run_names, results

if __name__ == '__main__':
    """
    data = {
        "default": {
            "exact": 0.03369199731002018,
            "exact_relaxed": 0.034364492266308,
            "near_relation": 0.08117014122394083
        },
        "relations": {
            "exact": 0.05,
            "exact_relaxed": 0.04,
            "near_relation": 0.09
        },
        "mask": {
            "exact": 0.06,
            "exact_relaxed": 0.6,
            "near_relation": 0.012
        }
    }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["lama", "commonsense_qa"], default="lama",
                        help="eval dataset")

    args = parser.parse_args()
    run_names, results_data = load_results_data(data_dir=f'eval/results/{args.dataset}')
    df_evals = pd.DataFrame(data=results_data, index=run_names)
    title = f'{args.dataset} evaluation'
    plot_results(title, df_evals, 'Accuracy (0-100%)', args.dataset)
    data_agglomeration = {str(run_name): results_data[i] for i, run_name in enumerate(run_names)}
    write_dict_2_json_file(data_agglomeration, f'{args.dataset}_eval.json', f'eval/plots/{args.dataset}')