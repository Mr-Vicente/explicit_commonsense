from typing import Dict
from transformers import IntervalStrategy, Seq2SeqTrainingArguments
from transformers.trainer_utils import EvaluationStrategy

from utils import get_device, read_json_file_2_dict, write_dict_2_json_file

class DefaultModelConfiguration:

    def __init__(self, configuration: Dict = None, model_type:str=''):
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
        self.per_device_train_batch_size = configuration.get("per_device_train_batch_size", 128)  # TODO
        # batch size for evaluation
        self.per_device_eval_batch_size = configuration.get("per_device_eval_batch_size", 128)  # TODO
        self.gradient_accumulation_steps = configuration.get("gradient_accumulation_steps", 1)
        self.warmup_steps = configuration.get("warmup_steps", 900)
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
        self.metric_for_best_model = configuration.get("metric_for_best_model", "combined")
        self.greater_is_better = configuration.get("greater_is_better", True)  # TODO set

        # Generation config
        self.generation_max_length = configuration.get("generation_max_length", 32)  # TODO
        self.generation_num_beams = configuration.get("generation_num_beams", 4)

        # output directory
        self.kg = configuration.get('kg', 'none')
        self.mask = configuration.get('mask', 'none')
        self.datasets = configuration.get('datasets', 'none')

        # run_name
        #run_name_start = f'{self.model_type}_{self.datasets}_{self.kg}_{self.mask}'
        self.run_name = configuration.get('run_name', self.get_run_name())
        idx = self.update_model_type_run_idx(self.run_name)
        self.run_name_idx = f'{self.run_name}_{idx}'

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
        self.report_lib = configuration.get("report_lib", ['wandb']) #comet_ml wandb
        self.device = get_device()

    def update_model_type_run_idx(self, run_name):
        runs_info = read_json_file_2_dict('runs_info.json', store_dir='runs')
        idx = runs_info.get(run_name, 0)
        new_idx = idx + 1
        runs_info[run_name] = new_idx
        write_dict_2_json_file(runs_info, 'runs_info.json', store_dir='runs')
        return idx

    def get_run_idx(self, run_name_start):
        runs_info = read_json_file_2_dict('runs_info.json', store_dir='runs')
        idx = runs_info.get(run_name_start, 0)
        return idx

    def get_run_name(self):
        run_name_start = f'{self.model_type}_{self.datasets}_{self.kg}_{self.mask}'
        idx = self.get_run_idx(run_name_start)
        run_name = f'{run_name_start}_{idx}_{self.per_device_train_batch_size}'
        return run_name

    def output_dir_from_params(self) -> str:
        run_name = self.run_name_idx
        out_dir_name = f'./trained_models/{run_name}'
        return out_dir_name

    def get_trainer_training_arguments(self, report=True, metric_for_best_model='combined') -> Seq2SeqTrainingArguments:
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
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=self.greater_is_better,
            generation_max_length=self.generation_max_length,
            generation_num_beams=self.generation_num_beams,
            fp16=True,
            predict_with_generate=True,
            remove_unused_columns=False,
            #debug="underflow_overflow"
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