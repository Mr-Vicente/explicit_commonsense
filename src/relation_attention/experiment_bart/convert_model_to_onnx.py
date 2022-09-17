
from pathlib import Path
import transformers
from transformers import PreTrainedModel
from transformers.utils.generic import TensorType

from custom_bart.bart_onnx import BartCustumOnnxConfig
from inference import RelationsInference
from utils import Model_Type, KGType

from inspect import signature

def ensure_model_and_config_inputs_match(model, model_inputs):
    """
    :param model_inputs: :param config_inputs: :return:
    """
    if issubclass(type(model), PreTrainedModel):
        forward_parameters = signature(model.forward).parameters
    else:
        forward_parameters = signature(model.call).parameters
    model_inputs_set = set(model_inputs)

    # We are fine if config_inputs has more keys than model_inputs
    forward_inputs_set = set(forward_parameters.keys())
    is_ok = model_inputs_set.issubset(forward_inputs_set)
    print('model_inputs_set:', model_inputs_set)
    print('forward_inputs_set:', forward_inputs_set)

    # Make sure the input order match (VERY IMPORTANT !!!!)
    matching_inputs = forward_inputs_set.intersection(model_inputs_set)
    ordered_inputs = [parameter for parameter in forward_parameters.keys() if parameter in matching_inputs]
    return is_ok, ordered_inputs

if __name__ == '__main__':
        #model_path = "/Users/mrvicente/Documents/Education/Thesis/code/pythonProject/models/bart-ra/checkpoint-2331"
        model_path = "/home/fm.vicente/data/models_weights/trained_models_3/facebook-bart-large_default_commongen_none_none_L-default_DS-default_wLearnEmb_2/checkpoint-2331"
        rels_inf = RelationsInference(model_path=model_path,
                           kg_type=KGType.CONCEPTNET,
                           model_type=Model_Type.RELATIONS,
                           max_length=32)
        tokenizer = rels_inf.tokenizer
        model = rels_inf.model
        bart_config = rels_inf.config

        onnx_config = BartCustumOnnxConfig(bart_config)

        model_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)
        inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
        print('Starting Export!')
        # export
        onnx_inputs, onnx_outputs = transformers.onnx.export(
                preprocessor=tokenizer,
                model=model,
                config=onnx_config,
                opset=13,
                output=Path("/home/fm.vicente/data/onnx/bart-ra.onnx")
        )
        print('done! :)')