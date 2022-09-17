#############################
#   Imports
#############################

# Python modules
from typing import Optional, Tuple, List, Dict
import os

# Remote modules
from evaluate import load
from transformers import (
    BartForConditionalGeneration,
    Seq2SeqTrainer, BartTokenizer,
    PreTrainedTokenizer
)

import numpy as np

# Local modules
from data.collator import BartDataCollator
from custom_tokenizer.bart_custom_tokenizer_fast import BartCustomTokenizerFast
from custom_bart.bart_for_conditional_generation import BartCustomForConditionalGeneration
from custom_bart.config import BartCustomConfig
from utils import get_device, Model_Type, Data_Type, Head_Mask, read_json_file_2_dict
from data.RelationsDatasets import (
    RelationsDataset,
    MaskRelationsDataset
)
from data.DatasetGeneral import DatasetGeneral
from data.data_preparation import load_and_preprocess_data
from run_config.configuration import DefaultModelConfiguration
from kgs_binding.kg_base_wrapper import KGBaseHandler
from kgs_binding.kg_qa_binding_utils import get_kg_qa_data_metadata, from_relations_path_2_relations, KGHandler_to_str
from model_utils import create_layers_head_mask

from trainer.custom_trainer import CustomSeq2SeqTrainer
from callbacks.custom_wandb_callback import CustomWandbCallback

from eval.custom_metric import compute_bleu_score


#############################
#   Constants
#############################
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#############################
#   Stuff
#############################

bleu_metric = load('bleu')
meteor_metric = load('meteor')
rouge_metric = load('rouge')
tokenizer: PreTrainedTokenizer = PreTrainedTokenizer()

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    blue_decoded_labels = [[label.strip()] for label in decoded_labels]
    decoded_labels = [label.strip() for label in decoded_labels]

    print("predictions: ", decoded_preds[0])
    print("references: ", decoded_labels[0])

    bleu_result = compute_bleu_score(bleu_metric, decoded_preds, blue_decoded_labels, max_order=3)
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_results = rouge_metric.compute(predictions=decoded_preds,references=decoded_labels, use_aggregator=False)
    rouge_results = rouge_results['rougeL']
    rouge_results = np.mean([s.fmeasure for s in rouge_results])
    metrics = {
        "bleu": bleu_result["bleu"] * 100.0,
        "meteor": meteor_result["meteor"] * 100.0,
        "rouge": rouge_results * 100.0,
    }
    metrics["combined"] = (metrics["bleu"] + metrics["meteor"] + metrics["rouge"]) / 3
    return metrics
"""
def compute_metrics(eval_preds):
    # Your previous metric computation
    metrics["combined"] = (metrics["accuracy"] + metrics["f1"]) / 2
    return metrics
"""

class Bart_QA:
    def __init__(self,  model_name: str = "facebook/bart-large",
                        experiment_type: Model_Type = Model_Type.DEFAULT,
                        kg_handler: KGBaseHandler=None,
                        heads_mask_type: Head_Mask = Head_Mask.ALL,
                        specific_heads=None,
                        run_config:Dict=None
                 ):
        global tokenizer
        self.device: str = get_device()
        self.run_config: DefaultModelConfiguration = DefaultModelConfiguration(run_config, experiment_type.value)
        self.config: Optional[BartCustomConfig] = None
        self.experiment_type = experiment_type
        self.kg_handler = kg_handler
        self.tokenizer, self.model = self.load_model(model_name, experiment_type,
                                                     kg_handler, heads_mask_type, specific_heads)
        tokenizer = self.tokenizer
        max_length: int = self.run_config.generation_max_length
        self.data_collator = BartDataCollator(tokenizer=self.tokenizer, model=self.model, max_length=max_length)

    def load_commonsense_tokenizer_and_model(self, model_name, kg_handler: KGBaseHandler,
                                             there_is_difference_between_relations, is_simple_mask_commonsense,
                                             heads_mask_type: Head_Mask = Head_Mask.ALL,
                                             specific_heads=None
                                             ):
        assert kg_handler is not None
        tokenizer = BartCustomTokenizerFast.from_pretrained(model_name)
        relation_names = kg_handler.get_relation_types()
        tokenizer.set_known_relation_names(relation_names)
        tokenizer.set_operation_mode(there_is_difference_between_relations=there_is_difference_between_relations)
        self.config = BartCustomConfig()
        self.config.num_relation_kinds = len(relation_names)
        self.config.is_simple_mask_commonsense = is_simple_mask_commonsense
        print('heads_mask_type:', heads_mask_type)
        heads_mask = create_layers_head_mask(self.config, heads_mask_type, specific_heads)
        self.config.heads_mask = heads_mask
        print('heads_mask:', heads_mask)
        self.run_config.kg = KGHandler_to_str(kg_handler)
        self.run_config.mask = heads_mask_type.value
        model = BartCustomForConditionalGeneration.from_pretrained(model_name, config=self.config)
        return tokenizer, model

    def load_model(self, model_name,
                   model_type: Model_Type=Model_Type.RELATIONS,
                   kg_handler: KGBaseHandler=None,
                   heads_mask_type: Head_Mask = Head_Mask.ALL,
                   specific_heads=None
                   ):
        print('-----Started loading model-----')
        if model_type == Model_Type.RELATIONS:
            tokenizer, model = self.load_commonsense_tokenizer_and_model(model_name, kg_handler,
                                                                         True, False,
                                                                         heads_mask_type, specific_heads)
        elif model_type == Model_Type.MASK:
            tokenizer, model = self.load_commonsense_tokenizer_and_model(model_name, kg_handler,
                                                                         False, True,
                                                                         heads_mask_type, specific_heads)
        elif model_type == Model_Type.DEFAULT:
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
        print('-----Ended loading model-----')
        return tokenizer, model

    def load_data(self,
                  dataset_types: List[Data_Type],
                  experiment_type=Model_Type.DEFAULT
        ):
        print('-----Started loading data-----')
        print(dataset_types[0])
        training_data, validation_data, test_data = load_and_preprocess_data(dataset_types, limit=5096)
        if experiment_type == Model_Type.DEFAULT:
            train_dataset = DatasetGeneral(training_data, tokenizer=self.tokenizer, device=self.device)
            validation_dataset = DatasetGeneral(validation_data, tokenizer=self.tokenizer, device=self.device)
            test_dataset = DatasetGeneral(test_data, tokenizer=self.tokenizer, device=self.device)
            return train_dataset, validation_dataset, test_dataset

        relations_metadata = get_kg_qa_data_metadata(self.kg_handler)
        relations_data = from_relations_path_2_relations(dataset_types, metadata=relations_metadata)
        if experiment_type == Model_Type.RELATIONS:
            train_dataset = RelationsDataset(training_data, tokenizer=self.tokenizer, device=self.device,
                                             relations_data=relations_data)
            validation_dataset = RelationsDataset(validation_data, tokenizer=self.tokenizer, device=self.device,
                                                  relations_data=relations_data)
            test_dataset = RelationsDataset(test_data, tokenizer=self.tokenizer, device=self.device,
                                            relations_data=relations_data)
        elif experiment_type == Model_Type.MASK:
            train_dataset = MaskRelationsDataset(training_data, tokenizer=self.tokenizer, device=self.device,
                                                 relations_data=relations_data)
            validation_dataset = MaskRelationsDataset(validation_data, tokenizer=self.tokenizer, device=self.device,
                                             relations_data=relations_data)
            test_dataset = MaskRelationsDataset(test_data, tokenizer=self.tokenizer, device=self.device,
                                             relations_data=relations_data)
        else:
            raise NotImplementedError()
        print('------Ended loading data------')
        return train_dataset, validation_dataset, test_dataset

    def train(self, train_dataset, valid_dataset, test_dataset, report=False):
        trainer = CustomSeq2SeqTrainer(
            model=self.model.to(self.device),  # the instantiated Transformers model to be trained
            tokenizer=tokenizer,
            args=self.run_config.get_trainer_training_arguments(report=report),  # get the training arguments
            train_dataset=train_dataset,  # training dataset
            eval_dataset=valid_dataset,  # evaluation dataset
            # metrics to compute
            compute_metrics=compute_metrics,
            data_collator=self.data_collator,
            #callbacks=[CustomWandbCallback()]
        )
        # actual training and validation procedure
        print("-------------------")
        print("Started Training...")
        train_results = trainer.train()
        print("-----Ended Training-----")
        print("Training Results:")
        print(train_results)

        # actual test set evaluation procedure
        evaluator = Seq2SeqTrainer(
            model=trainer.model.to(self.device),  # it loads the best model at the end of training
            tokenizer=tokenizer,
            args=self.run_config.get_trainer_testing_arguments(report=report),
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator,
            #callbacks=[EarlyStoppingCallback()]
        )

        # evaluate the best model from training on the test set
        print("-------------------")
        print("Started Testing...")
        test_results = evaluator.evaluate()
        print("-----Testing-----")
        print("Test Results:")
        print(test_results)