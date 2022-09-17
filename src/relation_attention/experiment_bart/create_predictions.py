#############################
#   Imports
#############################

# Python modules
import argparse
import csv
# Remote modules

# Local modules
from inference import Inference, RelationsInference
from utils import (
    Model_Type,
    KGType,
    read_txt_2_list
)

#############################
#   helper funcs
#############################
def get_args():
    print('-----Argument parsing------', flush=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--default_commongen_models_paths", nargs='+', type=str, default=None, help="")
    parser.add_argument("--relations_commongen_models_paths", nargs='+', type=str, default=None, help="")
    parser.add_argument("--default_absqa_models_paths", nargs='+', type=str, default=None, help="")
    parser.add_argument("--relations_absqa_models_paths", nargs='+', type=str, default=None, help="")

    args = parser.parse_args()
    return args

#############################
#  helper functions
#############################

def infer(inference_model, model_path, limit=100):
    model_path = '/'.join(model_path.split('/')[:-1])
    if 'commongen' in model_path:
        all_input_data = read_txt_2_list('concepts.txt', f'/home/fm.vicente/explicit_commonsense/src/mturk')[:limit]
    else:
        all_input_data = read_txt_2_list('input.txt', f'{model_path}')[:limit]
    #all_label_data = read_txt_2_list('gold.txt', f'{model_path}')[:limit]
    """
    all_input_data = [
        "what is the meaning of life?",
        "what is a cactus?",
        "which is faster, a car or a snail?",
        "what is the healthiest way to cook meat?",
        "are we living in the matrix",
        "is art meaningless?",
        "how many legs do insects have?",
        "do you like cats?",
        "are dog people better than cat people?"
    ]
    """
    all_input_data = [
        "why india is the largest producer of butter?",
        "why the cacao bean is native to mexico and both central and south america?",
        "why ancient Greeks used ovens mainly to make bread?",
        "why spices are highly rich in antioxidant?",
        "why china is the world’s largest carrot producer?",
        "why theoretically, wine can be used to fuel a car?",
        "why the iconic chef's hat is called a toque?",
        "why the fear of cooking is called 'mageirocophobia'?",
        "why microwaving is the healthiest way to cook vegetables?",
        "why candy floss was invented by a dentist?",
        "why popcorn was the first food to be microwaved?",
        "why each vegan spares 30 animal lives a year?"
    ]
    all_label_data = ["" for _ in all_input_data]
    #with open(f'{model_path}/predictions.csv', 'w') as f:
    #    predictions_writer = csv.writer(f)
    for i,(input_data, label_data) in enumerate(zip(all_input_data, all_label_data)):
        #print(i,': ', (input_data, label_data), flush=True)
        if 'constraint' in model_path:
            response = inference_model.generate_contrained_based_on_context([input_data])
        else:
            response = inference_model.generate_based_on_context(input_data)
        generation = response[0][0]
        print(i,': ', (input_data, generation), flush=True)
            #predictions_writer.writerow([input_data, generation, label_data])
    del inference_model


if __name__ == "__main__":
    print('[Started]', flush=True)
    args = get_args()

    default_commongen_models_paths = args.default_commongen_models_paths
    relations_commongen_models_paths = args.relations_commongen_models_paths
    default_absqa_models_paths = args.default_absqa_models_paths
    relations_absqa_models_paths = args.relations_absqa_models_paths

    print('[Args loaded]', flush=True)

    """
    print('==default_commongen_models_paths===\n', flush=True)
    for model_path in default_commongen_models_paths:
        print('Currently: ', model_path, flush=True)
        inf_model = Inference(model_path, max_length=32)
        infer(inf_model, model_path)

    print('==relations_commongen_models_paths===\n', flush=True)
    for model_path in relations_commongen_models_paths:
        print('Currently: ', model_path, flush=True)
        inf_model = RelationsInference(model_path=model_path,
                                      kg_type=KGType.CONCEPTNET,
                                      model_type=Model_Type.RELATIONS,
                                      max_length=32)
        infer(inf_model, model_path)

    """
    print('==default_absqa_models_paths===\n', flush=True)
    for model_path in default_absqa_models_paths:
        print('Currently: ', model_path, flush=True)
        inf_model = Inference(model_path, max_length=128)
        infer(inf_model, model_path)

    print('==relations_absqa_models_paths===\n', flush=True)
    for model_path in relations_absqa_models_paths:
        print('Currently: ', model_path, flush=True)
        inf_model = RelationsInference(model_path=model_path,
                                       kg_type=KGType.CONCEPTNET,
                                       model_type=Model_Type.RELATIONS,
                                       max_length=128)
        infer(inf_model, model_path)
