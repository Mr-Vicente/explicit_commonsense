
#############################
#   Imports
#############################

# Python modules
import sys
from typing import List
import argparse
# Remote modules
from evaluate import load
from transformers import (
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    BartTokenizer,
)
import numpy as np
## performance and logging
import wandb
# Local modules
from run_config.configuration import DefaultModelConfiguration
from run_config.pretrain_configuration import PretrainModelConfiguration
from utils import (
    KGType,
    Data_Type,
    read_json_file_2_dict,
    get_device
)
from data.data_preparation import load_csv_data, split_data
from data.pretrain_dataset import PretrainDataset
from kgs_binding.kg_base_wrapper import KGBaseHandler
from kgs_binding.kg_qa_binding_utils import load_kg_handler

#############################
#   Constants
#############################

if __name__ == '__main__':
    print('-----Argument parsing------')
    parser = argparse.ArgumentParser()

    parser.add_argument("--knowledge", type=str, choices=["conceptnet", "swow", "cskg"], default=None, help="uses knowledge")
    args = parser.parse_args()

    # pre_processing_args (turn strs into enums)
    print('knowledge:', args.knowledge)

    print('-----Weights and biases init------')
    idx = read_json_file_2_dict('pretrain_info.json', store_dir='runs').get("run_idx", 0)
    run_name = f'pretrain_{args.knowledge}_{idx}'
    """
    wandb.init(project=f"{run_name}",
               entity="mr-vicente",
               name=run_name)
    print(f'WandB initiated: {run_name}')
    """

    print('-----Starting BART------')
    model_name = 'facebook/bart-large'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # get available device
    device: str = get_device()

    run_config: PretrainModelConfiguration = PretrainModelConfiguration({
        'kg': args.knowledge
    }, 'pretrain_bart', run_name)
    # Load and prepare data
    filename = f'{args.knowledge}_bart.txt'
    data = load_csv_data(data_path=f'kgs_binding/bart_input/{args.knowledge}_bart.txt')
    train_data, val_data, test_data = split_data(data, val_percentage=0.05, test_percentage=0.05)
    print('train:', len(train_data), 'val:', len(val_data), 'test:', len(test_data))
    train_dataset = PretrainDataset(train_data, Data_Type.CONCEPTNET, tokenizer)
    validation_dataset = PretrainDataset(val_data, Data_Type.CONCEPTNET, tokenizer)
    test_dataset = PretrainDataset(test_data, Data_Type.CONCEPTNET, tokenizer)

    #define metrics
    accuracy_metric = load("accuracy")


    def calc_accuracy(predictions, references):
        tmp = []
        for pred, ref in zip(predictions, references):
            pred_diff = len(pred) - len(ref)
            if pred_diff > 0:  # pred biggest
                for i in range(pred_diff):
                    ref.append(1)
            elif abs(pred_diff) > 0:  # ref biggest
                for i in range(abs(pred_diff)):
                    pred.append(1)
            acc = accuracy_metric.compute(predictions=pred, references=ref)
            tmp.append(acc.get("accuracy", -1) * 100)
        mean_acc = np.mean(tmp)
        return {"accuracy": mean_acc}


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.roll(preds, -1, axis=1)
        preds = preds.transpose()
        preds[:][-1] = 1
        preds = preds.transpose()
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("predictions: ", decoded_preds[0])
        print("references: ", decoded_labels[0])

        preds = tokenizer(decoded_preds, add_special_tokens=False).input_ids
        labels = tokenizer(decoded_labels, add_special_tokens=False).input_ids
        # Some simple post-processing
        accuracy_result = calc_accuracy(predictions=preds, references=labels)
        metrics = {
            "accuracy": accuracy_result["accuracy"],
        }
        return metrics

    # Further Pretrain bart
    trainer = Seq2SeqTrainer(
        model=model.to(device),  # the instantiated Transformers model to be trained
        tokenizer=tokenizer,
        args=run_config.get_trainer_training_arguments(report=False),  # get the training arguments
        train_dataset=train_dataset,  # training dataset
        eval_dataset=validation_dataset,  # evaluation dataset
        # metrics to compute
        compute_metrics=compute_metrics,
        #data_collator=self.data_collator,
        # callbacks=[CustomWandbCallback()],
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
        model=model.to(device),  # it loads the best model at the end of training
        tokenizer=tokenizer,
        args=run_config.get_trainer_testing_arguments(report=False),
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        #data_collator=self.data_collator,
        # callbacks=[EarlyStoppingCallback()]
    )
    # evaluate the best model from training on the test set
    print("-------------------")
    print("Started Testing...")
    test_results = evaluator.evaluate()
    print("-----Testing-----")
    print("Test Results:")
    print(test_results)

