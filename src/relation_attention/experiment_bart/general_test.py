#############################
#   Imports
#############################

# Python modules

# Remote modules

# Local modules
from inference import Inference, RelationsInference
from utils import Model_Type, KGType

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class GeneralInference:
    models = {
        'relations': './trained_models/relations_commongen_conceptnet_none_1_3e-05_128/checkpoint-5170',
        'mask': './trained_models/relation_facebook-bart-large_3e-05_16/checkpoint-197220',
        'default': './trained_models/default_commongen_none_none_2_3e-05_256/checkpoint-2590/',
    }
    def __init__(self, model_type: Model_Type, kg_graph_str='conceptnet', max_length=32):
        self.model_type = model_type
        self.inference = self.load_inference(model_type, kg_graph_str, max_length)

    def load_inference(self, model_type, kg_graph_str='conceptnet', max_length=32):
        model_type_str = model_type.value
        model_path = GeneralInference.models[model_type_str]
        if model_type == Model_Type.DEFAULT:
            return Inference(model_path, max_length=max_length)

        kg = KGType(kg_graph_str)
        if model_type == Model_Type.MASK:
            return RelationsInference(model_path=model_path,
                                      kg_type=kg,
                                      model_type=Model_Type.MASK,
                                      max_length=max_length)
        elif model_type == Model_Type.RELATIONS:
            return RelationsInference(model_path=model_path,
                                      kg_type=KGType.CONCEPTNET,
                                      model_type=Model_Type.RELATIONS,
                                      max_length=max_length)



if __name__ == '__main__':
    QUESTION = "What is the opposite of being sad?"

    relation_inf = GeneralInference(
        model_type=Model_Type.RELATIONS
    )
    relation_answer = relation_inf.inference.answer_question(QUESTION)
    #default_inf = GeneralInference(
    #    model_type=Model_Type.DEFAULT
    #)
    #default_answer = default_inf.inference.answer_question(QUESTION)

    print('Relations Answer:', relation_answer)
    #print('Default Answer:', default_answer)