#############################
#   Imports
#############################

# Python modules
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import os
import sys
import time
from packaging import version

# Remote modules
from tqdm import tqdm
import torch.distributed as dist

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import Seq2SeqTrainer, TrainingArguments, LogitsProcessorList
from transformers.integrations import (  # isort: split
    hp_params,
    is_fairscale_available,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
)
from transformers.trainer import (
    logger,
)
from transformers.trainer_utils import (
    EvalPrediction,
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    set_seed,
    speed_metrics,
)
from transformers.utils import (
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_apex_available,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_reinit, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.optimization import Adafactor, get_scheduler

import warnings

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_apex_available():
    import apex.amp as amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    #from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    #from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    #from fairscale.nn.wrap import auto_wrap
    #from fairscale.optim import OSS
    #from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

# Local modules
from utils import LossType, inputs_introspection_print, Model_Type, ScoringType
from trainer.custom_losses import (
    calc_cp_def_loss,
    calc_cp_rp_def_loss,
    calc_prp_nrp_def_loss,
    calc_def_constraint_loss
)
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder
from heads_analysis.heads_surpervisor import HeadsSupervisor
from custom_bart.custom_logits_processor import KGLogitsProcessor, MultipleChoiceProcessor

from transformers.debug_utils import DebugOption
from codecarbon import EmissionsTracker

#############################
#   Constants
#############################

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

#############################
#   Helper fucntions
#############################

#############################
#   Stuff
#############################


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        custom_loss_type:LossType=LossType.DEFAULT,
        heads_supervisor:Optional[HeadsSupervisor]=None,
        relation_mapper_builder:RelationsMapperBuilder=None,
        kg_decoding_scoring_type:ScoringType=LossType.DEFAULT,
        run_dir:str='.'
    ):
        super(CustomSeq2SeqTrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.relation_mapper_builder = relation_mapper_builder
        self.custom_loss_type = custom_loss_type
        self.counter_op = 0
        self.heads_supervisor:HeadsSupervisor = heads_supervisor
        self.kg_decoding_scoring_type = kg_decoding_scoring_type
        proj_name = run_dir.split('/')[-1]
        if proj_name in ['.', '', None]:
            proj_name='codecarbon'
        self.emissions_tracker = EmissionsTracker(project_name=proj_name, output_dir=run_dir)

    """
        def compute_loss(self, model, inputs, return_outputs=False, alpha=1.0):
        original_rels_amount = get_original_rels_amount(inputs)
        input_ids = inputs.get('input_ids')
        inputs['reduce_ce']=False
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        logits = outputs.logits
        masked_index = input_ids >= 3  # ignore special_tokens
        penalize_batch_preds = []
        # relevant_ids = relevant_ids[masked_index]
        for b, b_logs in enumerate(logits):
            probs = b_logs[masked_index[b]].softmax(dim=0)
            _values, predictions = probs.topk(1)
            gen_phrase = self.tokenizer.decode(predictions.squeeze(1))
            reals_info = self.process_context(gen_phrase)
            score = score_based_on_amount_of_relations(reals_info, original_rels_amount=original_rels_amount)
            del reals_info
            penalize_batch_preds.append(score)
        # Resize and average loss per sample
        #print('loss:', loss)
        #print('loss.shape:', loss.shape)
        loss_per_sample = loss.view(logits.size(0), logits.size(1)).mean(axis=1)
        # Calculate and scale weighting
        weights = torch.tensor(penalize_batch_preds).to(model.device)
        weights = alpha * (1.0 + weights)
        # Calculate weighted average
        loss = (loss_per_sample * weights).mean()
        return (loss, outputs) if return_outputs else loss
    """
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        original_rels_amount = get_original_rels_amount(inputs)
        #print('original_rels_amount:', original_rels_amount)
        input_ids = inputs.get('input_ids')
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        logits = outputs.logits
        masked_index = input_ids >= 3  # ignore special_tokens
        penalize_batch_preds = []
        # relevant_ids = relevant_ids[masked_index]
        #print('logits.shape:', logits.shape)
        for b, b_logs in enumerate(logits):
            probs = b_logs[masked_index[b]].softmax(dim=0)
            _values, predictions = probs.topk(1)
            gen_phrase = self.tokenizer.decode(predictions.squeeze(1))
            #print('gen_phrase:', gen_phrase)
            reals_info = self.process_context(gen_phrase)
            score = score_based_on_amount_of_relations(reals_info, original_rels_amount=original_rels_amount[b])
            del reals_info
            penalize_batch_preds.append(score)
        # Resize and average loss per sample
        #print('loss:', loss)
        #print('loss.shape:', loss.shape)
        # Calculate and scale weighting
        scoring_weights = torch.tensor(penalize_batch_preds).to(model.device)
        scoring_weights = scoring_weights.mean()
        # Calculate weighted average
        #print('loss:::', loss)
        #print('scoring_weights:::', scoring_weights)
        loss = loss + scoring_weights
        return (loss, outputs) if return_outputs else loss
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.heads_supervisor and self.heads_supervisor.is_updating():
            inputs['output_attentions'] = False
        if self.counter_op == 0:
            inputs_introspection_print(self.tokenizer, inputs)
            self.counter_op=1
        if self.kg_decoding_scoring_type==ScoringType.CONSTRAINT:
            loss_f = super().compute_loss
            loss, outputs = calc_def_constraint_loss(loss_f, inputs, model,
                                                     self.tokenizer, return_outputs=True)
        elif self.custom_loss_type == LossType.DEFAULT:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        elif self.custom_loss_type == LossType.CP_DEF:
            loss_f = super().compute_loss
            loss, outputs = calc_cp_def_loss(loss_f, inputs, model,return_outputs=True)
        elif self.custom_loss_type == LossType.PRP_NRP_DEF:
            loss_f = super().compute_loss
            loss, outputs = calc_prp_nrp_def_loss(loss_f, inputs, model,
                                                self.tokenizer, self.relation_mapper_builder,
                                                return_outputs=True)
        elif self.custom_loss_type == LossType.CP_RP_DEF:
            loss_f = super().compute_loss
            loss, outputs = calc_cp_rp_def_loss(loss_f, inputs, model,
                                                self.tokenizer, self.relation_mapper_builder,
                                                return_outputs=True)
        else:
            raise NotImplementedError()
        if self.heads_supervisor and self.heads_supervisor.is_updating():
            #print('outputs:', outputs)
            self.heads_supervisor.store_info_head_supervisors(outputs)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if self.heads_supervisor:
            #print("here.....")
            #print("self.heads_supervisor.heads_mask.grad:", self.heads_supervisor.heads_mask.grad)
            inputs["head_mask"] = self.heads_supervisor.heads_mask
        inputs = self._prepare_inputs(inputs)
        #if self.heads_supervisor and self.heads_supervisor.still_updating():
        #    inputs["head_mask"].requires_grad_(requires_grad=True)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        #print('t1:', inputs["head_mask"][0])
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        #print('t2:', inputs["head_mask"][0])

        #### Insert Supervisor stuff
        if self.heads_supervisor and self.heads_supervisor.is_updating():
            self.heads_supervisor.update_head_supervisors(inputs)
        return loss.detach()

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None
        tot_co2_emission = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        self.emissions_tracker.start()
        for epoch in range(epochs_trained, num_train_epochs):
            if self.heads_supervisor and epoch == 1:
                self.heads_supervisor.evaluate_heads()
                self.heads_supervisor.reset_for_new_epoch()
                self.heads_supervisor.stop_updating()
                # Make sure the new heads get persistently stored at the end of training
                self.model.config.heads_mask = self.heads_supervisor.heads_to_vec()

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )


                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
            #emissions = self.emissions_tracker.stop()
            #self.log_co2(emissions)
            #tot_co2_emission += emissions

        emissions = self.emissions_tracker.stop()
        tot_co2_emission += emissions
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        logging_material = {**metrics, "total_co2_kg": tot_co2_emission}
        self.log(logging_material)
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def log_co2(self, co2:float) -> None:
        """
        Log `logs` on the various objects watching training.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        logs = {}
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{'epoch_co2_kg': co2}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)
        if "input_commonsense_relations" in inputs:
            gen_kwargs["relation_inputs"] = inputs.get("input_commonsense_relations", None)
        if self.heads_supervisor:
            gen_kwargs["head_mask"] = self.heads_supervisor.heads_mask
        gen_kwargs["no_repeat_ngram_size"] = 2
        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        #contexts = self.tokenizer.batch_decode(generation_inputs, skip_special_tokens=False)
        #generated_tokens = self.model.cs_generate(
        #    generation_inputs,
        #    contexts=contexts,
        #    model_input=inputs,
        #    use_kg=True,
        #    relation_mapper_builder=self.relation_mapper_builder,
        #    tokenizer=self.tokenizer,
        #    **gen_kwargs,
        #)
        if self.kg_decoding_scoring_type in [ScoringType.INTERPOL,ScoringType.MAX_PROB,ScoringType]:
            kg_logits_processor = KGLogitsProcessor(self.tokenizer, self.relation_mapper_builder)
            custom_logits_processor = LogitsProcessorList([kg_logits_processor]) #gets added to the default ones
        elif self.kg_decoding_scoring_type == ScoringType.MULTIPLE_CHOICE:
            input_ids_s = inputs['input_ids']
            choices_ids_s = []
            for input_ids in input_ids_s:
                eos_indexes = (input_ids == 2).nonzero(as_tuple=True)[0]
                eos_indexes = eos_indexes.tolist()
                choices_ids_s.append(input_ids[eos_indexes[0] + 1:eos_indexes[1]])
            #gen_kwargs['choices_ids'] = choices_ids_s
            custom_logits_processor = LogitsProcessorList([MultipleChoiceProcessor(self.tokenizer, choices_ids_s)])
        else:
            custom_logits_processor = LogitsProcessorList()
        if self.kg_decoding_scoring_type in [ScoringType.CONSTRAINT]:
            input_ids_s = inputs['input_ids']
            #print('input_ids_s:', input_ids_s)
            neighbours_contexts = []
            for input_ids in input_ids_s:
                eos_indexes = (input_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                bos_indexes = (input_ids == self.tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
                eos_indexes = eos_indexes.tolist()
                bos_indexes = bos_indexes.tolist()
                concepts_ids = input_ids[bos_indexes[-1] + 1:eos_indexes[-1]]
                concepts_str = self.tokenizer.decode(concepts_ids, skip_special_tokens=True)
                concepts_str_list = concepts_str.split(' ')
                neighbours_contexts.append(concepts_str_list)
            print('neighbours_contexts[0]:', neighbours_contexts[0])
            generated_tokens = self.model.cs_simple_generate(
                generation_inputs,
                neighbours_contexts=neighbours_contexts,
                model_input=inputs,
                use_kg=True,
                relation_mapper_builder=self.relation_mapper_builder,
                tokenizer=self.tokenizer,
                **gen_kwargs,
            )
        else:
            generated_tokens = self.model.generate(
                generation_inputs,
                logits_processor=custom_logits_processor,
                **gen_kwargs,
            )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        # Print eval
        self.print_eval(inputs["input_ids"], generated_tokens, inputs["labels"], max_number=2, ignore_special_tokens=True)

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # PRINT RESULTS
        self.print_eval(None, output.predictions, output.label_ids)

        return output.metrics

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def print_eval(self, input_ids, predictions, label_ids, max_number=10, ignore_special_tokens=False):
        print(f'########### EVALUATION ###########')
        if input_ids is not None:
            inputs_text = self.tokenizer.batch_decode(input_ids[:max_number], skip_special_tokens=False)
        else:
            inputs_text = [None for _ in range(max_number)]
        preds_text = self.tokenizer.batch_decode(predictions[:max_number], skip_special_tokens=ignore_special_tokens)
        labels_text = self.tokenizer.batch_decode(label_ids[:max_number], skip_special_tokens=ignore_special_tokens)
        for input_text, pred_text, label_text in zip(inputs_text, preds_text, labels_text):
            print('Input_text:', input_text, ' | ','Gen:', pred_text, ' |vs| ', 'Gold:', label_text)
        print('########### ----------- ###########')

