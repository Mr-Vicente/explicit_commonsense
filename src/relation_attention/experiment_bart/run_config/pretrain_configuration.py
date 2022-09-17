from typing import Dict
from transformers import IntervalStrategy, Seq2SeqTrainingArguments
from transformers.trainer_utils import EvaluationStrategy

from utils import get_device, read_json_file_2_dict, write_dict_2_json_file

class PretrainModelConfiguration:

    def __init__(self, configuration: Dict = None, model_type:str='', run_name:str=''):
        # model and tokenizer
        self.model_type = model_type
        if not configuration:
            configuration = {}
        self.suffix = configuration.get("suffix", "")  # way to distinguish between identical models
        self.model_name = configuration.get("model_name", "facebook/bart-large")
        self.tokenizer_name = configuration.get("tokenizer_name", "facebook/bart-large")
        self.do_lower_case = configuration.get("do_lower_case", True)

        # training and eval params
        self.num_train_epochs = configuration.get("num_train_epochs", 10)  # total number of training epochs  # TODO
        # batch size per device during training
        self.per_device_train_batch_size = configuration.get("per_device_train_batch_size", 64)  # TODO
        # batch size for evaluation
        self.per_device_eval_batch_size = configuration.get("per_device_eval_batch_size", 64)  # TODO
        self.gradient_accumulation_steps = configuration.get("gradient_accumulation_steps", 1)
        self.warmup_steps = configuration.get("warmup_steps", 1000)
        self.learning_rate = configuration.get("learning_rate", 3e-5)
        self.weight_decay = configuration.get("weight_decay", 0.01)  # strength of weight decay
        self.logging_dir = configuration.get("logging_dir", './logs')  # directory for storing logs
        self.logging_steps = configuration.get("logging_steps", 100)
        self.evaluation_strategy = configuration.get("evaluation_strategy", EvaluationStrategy.EPOCH.value)
        self.save_strategy = configuration.get("save_strategy", IntervalStrategy.EPOCH.value)
        # self.save_steps = configuration.get("save_steps",200 if self.use_small_dataset else 2000)
        self.save_total_limit = configuration.get("save_total_limit", 1)
        self.no_cuda = configuration.get("no_cuda", False)
        self.seed = configuration.get("seed", 42)
        self.metric_for_best_model = configuration.get("metric_for_best_model", "accuracy")
        self.greater_is_better = configuration.get("greater_is_better", True)  # TODO set

        # Generation config
        self.generation_max_length = configuration.get("generation_max_length", 128)  # TODO
        self.generation_num_beams = configuration.get("generation_num_beams", 4)

        # output directory
        self.kg = configuration.get("kg", 'none')
        self.mask = 'none'
        self.run_name = run_name
        start_run_name = f'pretrain_{self.kg}'
        self.update_model_type_run_idx(start_run_name)
        #self.train_run_name = configuration.get("train_run_name", f'train_{self.get_run_name()}')
        #self.test_run_name = configuration.get("test_run_name", f'test_{self.get_run_name()}')
        #self.output_dir = configuration.get("output_dir", self.output_dir_from_params())

        # Commonsense relations info
        self.swow_data_path = configuration.get("swow_data_path", (
            '', ''
        ))
        self.conceptnet_data_path = configuration.get("conceptnet_data_path", {
            'eli5': ('eli5_conceptnet_relation_data.json', '/home/fm.vicente/explicit_commonsense/src/relation_attention/experiment_bart/kgs_binding/model_kgs/conceptnet'),
            'stackexchange_qa': ('stackexchange_qa_conceptnet_relation_data.json', '/home/fm.vicente/explicit_commonsense/src/relation_attention/experiment_bart/kgs_binding/model_kgs/conceptnet')
        })
        self.cskg_data_path = configuration.get("cskg_data_path", (
            '', ''
        ))
        self.report_lib = configuration.get("report_lib", ['comet_ml']) #cometml wandb
        self.device = get_device()

    def update_model_type_run_idx(self, start_run_name):
        runs_info = read_json_file_2_dict('pretrain_info.json', store_dir='runs')
        idx = runs_info.get(start_run_name, 0)
        new_idx = idx + 1
        runs_info[start_run_name] = new_idx
        write_dict_2_json_file(runs_info, 'pretrain_info.json', store_dir='runs')
        return idx

    def output_dir_from_params(self) -> str:
        out_dir_name = f'./trained_models/{self.run_name}'
        return out_dir_name

    def get_trainer_training_arguments(self, report=True) -> Seq2SeqTrainingArguments:
        self.train_run_name = f'train_{self.run_name}'
        self.output_dir = self.output_dir_from_params()
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            # save_steps=training_configuration.save_steps,
            save_total_limit=self.save_total_limit,
            no_cuda=self.no_cuda,
            seed=self.seed,
            run_name=self.train_run_name,
            load_best_model_at_end=False,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            generation_max_length=self.generation_max_length,
            generation_num_beams=self.generation_num_beams,
            fp16=True,
            predict_with_generate=True,
            remove_unused_columns=False,
        )
        if report:
            training_args.report_to = self.report_lib
        return training_args

    def get_trainer_testing_arguments(self, report=True) -> Seq2SeqTrainingArguments:
        self.test_run_name = f'test_{self.run_name}'
        self.output_dir = self.output_dir_from_params()
        test_arguments = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,  # output directory
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            # batch size for evaluation
            logging_dir=self.logging_dir,  # directory for storing logs
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            no_cuda=False,
            seed=self.seed,
            run_name=self.test_run_name,
            generation_max_length=self.generation_max_length,
            generation_num_beams=self.generation_num_beams,
            fp16=True,
            predict_with_generate=True,
            remove_unused_columns=False
        )
        if report:
            test_arguments.report_to = self.report_lib
        return test_arguments