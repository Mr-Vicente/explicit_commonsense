#############################
#   Imports
#############################

# Python modules
import argparse
import os
import csv
from shutil import rmtree
from ast import literal_eval

# Remote modules
import wandb

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

def init_wandb(run_name=None):
    print('-----Weights and biases init------')
    wandb.init(project=f"bart_metrics_reports",
               entity="mr-vicente",
               name=run_name)
    print(f'WandB initiated: {run_name}')

def get_args():
    print('-----Argument parsing------')
    parser = argparse.ArgumentParser()

    parser.add_argument("--models_path", type=str, default="trained_models", help="uses knowledge")
    parser.add_argument("--keywords", nargs='+', type=str, default=None, help="model type") # choices=["wExtraRelations", "wContext", "wLearnEmb", etc]

    args = parser.parse_args()
    return args

def file_exists(path):
    return os.path.exists(path)

if __name__ == '__main__':
    # Get Script Parameters
    args = get_args()

    # Initialize Third party data dump provider
    #init_wandb(run_name=None)

    with open('reports/report.csv', 'w') as report_w:
        for experiment_path in os.listdir(args.models_path):
            if args.keywords:
                no_keyword = False
                for keyword in args.keywords:
                    if keyword not in experiment_path:
                        no_keyword=True
                if no_keyword:
                    continue
            # succesfully experiment is to report
            final_experiment_path = f'{args.models_path}/{experiment_path}'
            metrics_file_path = f'{final_experiment_path}/best.csv'
            if not file_exists(metrics_file_path):
                # rmtree(final_experiment_path) # to clean other empty experiments
                continue
            with open(f'{metrics_file_path}', 'r') as f:
                data = csv.reader(f)
                header = data.__next__()
                record = data.__next__()
                record = [literal_eval(r) for r in record]
                #print(header, '\n', record)
                header.insert(0, 'experiment')
                record.insert(0, experiment_path)
                csv_data_dict = dict(zip(header, record))
                report_writer = csv.writer(report_w)
                report_writer.writerows([header, record])
                #wandb.log_artifact({'experiment': experiment_path, **csv_data_dict})




    # Close API session
    #wandb.finish()

    print('----- All done :)))) ------')