#############################
#   Imports
#############################

# Python modules
from typing import Optional, Tuple, List, Dict
import itertools
import csv

# Remote modules
from transformers import (
    BartForConditionalGeneration,
    Seq2SeqTrainer, BartTokenizer,
    PreTrainedTokenizer, BartConfig
)

# Local modules
from data.collator import BartDataCollator
from custom_tokenizer.bart_custom_tokenizer_fast import BartCustomTokenizerFast
from custom_bart.bart_for_conditional_generation import BartCustomForConditionalGeneration
from custom_bart.config import BartCustomConfig, BartSmallCustomConfig
from kgs_binding import RelationsMapperBuilder
from utils import (
    get_device,
    Model_Type,
    Data_Type,
    Head_Mask,
    read_json_file_2_dict,
    tok_data_2_text,
    create_directory
)
from data.relations_datasets_preprocessing import (
    DefaultDataset,
    RelationsDataset,
    MaskRelationsDataset
)
from data.DatasetGeneral import DatasetGeneral
from data.datasets_model_handling import DatasetParsingUtils
from data.data_preparation import load_and_preprocess_data, load_tok_data, load_tok_data_simple
from run_config.configuration import DefaultModelConfiguration
from kgs_binding.kg_base_wrapper import KGBaseHandler
from kgs_binding.conceptnet_handler import ConceptNetHandler
from kgs_binding.kg_qa_binding_utils import get_kg_qa_data_metadata, from_relations_path_2_relations, KGHandler_to_str
from model_utils import create_layers_head_mask

from eval.metric_scorer import MetricScorer
from trainer.custom_trainer import CustomSeq2SeqTrainer
from heads_analysis.heads_surpervisor import HeadsSupervisor
from callbacks.custom_wandb_callback import CustomWandbCallback

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

simple_bart_tokenizer: BartTokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer: PreTrainedTokenizer = PreTrainedTokenizer()

class Bart_QA:
    def __init__(self,  model_name: str = "facebook/bart-large",
                        experiment_type: Model_Type = Model_Type.DEFAULT,
                        kg_handler: KGBaseHandler=None,
                        heads_mask_type: Head_Mask = Head_Mask.ALL,
                        specific_heads=None,
                        run_config:Dict=None,
                        args=None
                 ):
        global tokenizer
        self.args = args
        self.device: str = get_device()
        self.run_config: DefaultModelConfiguration = DefaultModelConfiguration(run_config, experiment_type.value)
        self.config: Optional[BartConfig] = None
        self.experiment_type = experiment_type
        # Even if default transform kg_handler into conceptnet
        kg_handler = ConceptNetHandler() if kg_handler is None else kg_handler
        self.kg_handler = kg_handler
        self.relation_mapper_builder = RelationsMapperBuilder(knowledge=self.kg_handler)
        self.tokenizer, self.model = self.load_model(model_name, experiment_type,
                                                     kg_handler, heads_mask_type, specific_heads)
        tokenizer = self.tokenizer
        self.max_length: int = self.run_config.generation_max_length
        self.data_collator = BartDataCollator(tokenizer=self.tokenizer, model=self.model, max_length=self.max_length)
        self.metric_scorer = MetricScorer(self.relation_mapper_builder)

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

        self.config = BartCustomConfig() if 'large' in self.args.pre_model else BartSmallCustomConfig()
        print('config_type:', self.config.__class__)
        self.config.should_embed_positions = self.args.learn_pos_embed_encoder
        print('learn_pos_embed_encoder:', self.config.should_embed_positions)
        self.config.num_relation_kinds = len(relation_names)
        self.config.is_simple_mask_commonsense = is_simple_mask_commonsense
        print('heads_mask_type:', heads_mask_type)
        heads_mask = create_layers_head_mask(self.config, heads_mask_type, specific_heads)
        self.config.heads_mask = heads_mask
        print('heads_mask:', heads_mask)
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
                  experiment_type=Model_Type.DEFAULT,
                  use_context_str='',
                  use_extra_rels_str='',
        ):
        print('-----Started loading data-----')
        print(dataset_types[0])
        first_data_type=dataset_types[0]
        datasets_str = Data_Type.data_types_to_str(dataset_types)

        if experiment_type == Model_Type.DEFAULT:
            store_dir = f'/home/fm.vicente/data/tok_data_simple'
            training_data = load_tok_data_simple('training', datasets=datasets_str, store_dir=store_dir, use_context_str=use_context_str)
            validation_data = load_tok_data_simple('validation', datasets=datasets_str, store_dir=store_dir,use_context_str=use_context_str)
            test_data = load_tok_data_simple('testing', datasets=datasets_str, store_dir=store_dir, use_context_str=use_context_str)

            train_dataset = DefaultDataset(training_data, device=self.device)
            validation_dataset = DefaultDataset(validation_data, device=self.device)
            test_dataset = DefaultDataset(test_data, device=self.device)
            #training_data, validation_data, test_data = load_and_preprocess_data(dataset_types)
            #training_dataset_parsing = DatasetParsingUtils(first_data_type, training_data, use_context=self.args.use_context)
            #train_dataset = DatasetGeneral(training_dataset_parsing,tokenizer=self.tokenizer, device=self.device, max_length=self.max_length)
            #validation_dataset_parsing = DatasetParsingUtils(first_data_type, validation_data, use_context=self.args.use_context)
            #validation_dataset = DatasetGeneral(validation_dataset_parsing,tokenizer=self.tokenizer, device=self.device, max_length=self.max_length)
            #test_dataset_parsing = DatasetParsingUtils(first_data_type, test_data, use_context=self.args.use_context)
            #test_dataset = DatasetGeneral(test_dataset_parsing, tokenizer=self.tokenizer, device=self.device, max_length=self.max_length)
            return train_dataset, validation_dataset, test_dataset

        kg_str = KGHandler_to_str(self.kg_handler)
        store_dir = f'/home/fm.vicente/data/tok_data'
        exp_str = self.experiment_type.value #'relations'
        training_data = load_tok_data('training', exp_type=exp_str, datasets=datasets_str, store_dir=store_dir, kg=kg_str, use_context_str=use_context_str, use_extra_rels_str=use_extra_rels_str)
        validation_data = load_tok_data('validation', exp_type=exp_str, datasets=datasets_str, store_dir=store_dir, kg=kg_str, use_context_str=use_context_str, use_extra_rels_str=use_extra_rels_str)
        test_data = load_tok_data('testing', exp_type=exp_str, datasets=datasets_str, store_dir=store_dir, kg=kg_str, use_context_str=use_context_str, use_extra_rels_str=use_extra_rels_str)

        # to compare with baselines afterwards
        run_path = self.run_config.output_dir_from_params()
        create_directory(run_path)
        with open(f'{run_path}/input.txt', 'w') as s, open(f'{run_path}/gold.txt', 'w') as g_s:
            test_input_text, test_labels_text = tok_data_2_text(tokenizer, test_data)
            for input_text, label in zip(test_input_text, test_labels_text):
                s.write(f'{input_text}\n')
                g_s.write(f'{label}\n')
        #

        if experiment_type == Model_Type.RELATIONS:
            train_dataset = RelationsDataset(training_data, device=self.device)
            validation_dataset = RelationsDataset(validation_data, device=self.device)
            test_dataset = RelationsDataset(test_data, device=self.device)
        elif experiment_type == Model_Type.MASK:
            train_dataset = MaskRelationsDataset(training_data, device=self.device)#,  limitation=2000)
            validation_dataset = MaskRelationsDataset(validation_data, device=self.device)
            test_dataset = MaskRelationsDataset(test_data, device=self.device)
        else:
            raise NotImplementedError()
        print('------Ended loading data------')
        return train_dataset, validation_dataset, test_dataset

    def compute_metrics_csqa(self, eval_preds):
        preds, labels = eval_preds
        # Some simple post-processing
        new_preds, new_labels, out_preds, out_labels = self.metric_scorer.prepare_csqa_text_generation(tokenizer, simple_bart_tokenizer, preds, labels)

        #metrics = self.metric_scorer.score_text_csqa(new_preds, new_labels)
        metrics = self.metric_scorer.score_text_csqa(out_preds, out_labels)
        return metrics

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        preds_text, labels_text, blue_labels_text = self.metric_scorer.preprocess_text_generation(tokenizer, preds, labels)

        print("predictions: ", preds_text[0])
        print("references: ", labels_text[0])

        metrics = self.metric_scorer.score_text_generation(preds_text, labels_text, blue_labels_text)
        return metrics

    def generate_w_test_data(self, evaluator: Seq2SeqTrainer, max_number=20):
        dataloader = evaluator.get_eval_dataloader()
        prediction_output = evaluator.prediction_loop(dataloader=dataloader, description="Post Generation")
        inputs_text = [self.tokenizer.batch_decode(input_data['input_ids']) for input_data in dataloader]
        inputs_text_limited = list(itertools.chain.from_iterable(inputs_text))[:max_number]
        predictions = prediction_output.predictions
        label_ids = prediction_output.label_ids
        preds_text = self.tokenizer.batch_decode(predictions[:max_number], skip_special_tokens=True)
        labels_text = self.tokenizer.batch_decode(label_ids[:max_number], skip_special_tokens=True)
        return inputs_text_limited, preds_text, labels_text

    def train(self, train_dataset, valid_dataset, test_dataset, loss_type, kg_decoding_scoring_type, report=False):
        run_path = self.run_config.output_dir_from_params()
        if 'commonsense_qa' in self.run_config.datasets:
            metrics_func = self.compute_metrics_csqa
            metric_for_best_model = 'accuracy'
        else:
            metrics_func = self.compute_metrics
            metric_for_best_model = 'combined'
        # just use heads supervision if it is intencional and with MASK model
        heads_supervisor = HeadsSupervisor(self.config.encoder_layers, self.config.encoder_attention_heads, get_device(verbose=False)) \
            if self.experiment_type == Model_Type.MASK and self.args.use_dynamic_heads else None
        #
        trainer = CustomSeq2SeqTrainer(
            model=self.model.to(self.device),  # the instantiated Transformers model to be trained
            tokenizer=tokenizer,
            args=self.run_config.get_trainer_training_arguments(
                report=report,
                metric_for_best_model=metric_for_best_model
            ),  # get the training arguments
            train_dataset=train_dataset,  # training dataset
            eval_dataset=valid_dataset,  # evaluation dataset
            # metrics to compute
            compute_metrics=metrics_func,
            data_collator=self.data_collator,
            #callbacks=[CustomWandbCallback()],
            custom_loss_type=loss_type,
            relation_mapper_builder=self.relation_mapper_builder,
            heads_supervisor=heads_supervisor,
            kg_decoding_scoring_type=kg_decoding_scoring_type,
            run_dir=run_path
        )
        # actual training and validation procedure
        print("-------------------")
        print("Started Training...")
        train_results = trainer.train()
        print("-----Ended Training-----")
        print("Training Results:")
        print(train_results)

        # actual test set evaluation procedure
        evaluator = CustomSeq2SeqTrainer(
            model=trainer.model.to(self.device),  # it loads the best model at the end of training
            tokenizer=tokenizer,
            args=self.run_config.get_trainer_testing_arguments(report=report),
            eval_dataset=test_dataset,
            compute_metrics=metrics_func,
            data_collator=self.data_collator,
            custom_loss_type=loss_type,
            relation_mapper_builder=self.relation_mapper_builder,
            heads_supervisor=heads_supervisor,
            kg_decoding_scoring_type=kg_decoding_scoring_type,
            run_dir=run_path
            #callbacks=[EarlyStoppingCallback()]
        )


        # evaluate the best model from training on the test set
        print("-------------------")
        print("Started Testing...")
        test_results = evaluator.evaluate()
        print("-----Testing-----")
        print("Test Results:")
        print(test_results)
        print("Writing CSV Results at:", run_path)
        with open(f'{run_path}/best.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(test_results.keys())
            writer.writerow(test_results.values())

        input_texts, predictions, labels = self.generate_w_test_data(evaluator=evaluator, max_number=20)
        with open(f'{run_path}/examples.csv', 'w') as f:
            writer = csv.writer(f)
            for input_text, pred, label in zip(input_texts, predictions, labels):
                writer.writerow([input_text, pred, label])
