#############################
#   Imports
#############################

# Python modules
import functools
from time import time
import math

# Remote modules
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class CustomTrainer:
    def __init__(self):
        pass

    def train_qa_s2s_epoch(self, model, dataset, tokenizer, optimizer, scheduler, args, e=0, curriculum=False):
        model.train()
        # make iterator
        if curriculum:
            train_sampler = SequentialSampler(dataset)
        else:
            train_sampler = RandomSampler(dataset)
        model_collate_fn = functools.partial(
            make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"
        )
        data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
        epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
        # accumulate loss since last print
        loc_steps = 0
        loc_loss = 0.0
        st_time = time()
        for step, batch_inputs in enumerate(epoch_iterator):
            pre_loss = model(**batch_inputs)[0]
            loss = pre_loss.sum() / pre_loss.shape[0]
            loss.backward()
            # optimizer
            if step % args.backward_freq == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            # some printing within the epoch
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0 or step == 1:
                print(
                    "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        e,
                        step,
                        len(dataset) // args.batch_size,
                        loc_loss / loc_steps,
                        time() - st_time,
                    )
                )
                loc_loss = 0
                loc_steps = 0


    def eval_qa_s2s_epoch(self, model, dataset, tokenizer, args):
        model.eval()
        # make iterator
        train_sampler = SequentialSampler(dataset)
        model_collate_fn = functools.partial(
            make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"
        )
        data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
        epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
        # accumulate loss since last print
        loc_steps = 0
        loc_loss = 0.0
        st_time = time()
        with torch.no_grad():
            for step, batch_inputs in enumerate(epoch_iterator):
                pre_loss = model(**batch_inputs)[0]
                loss = pre_loss.sum() / pre_loss.shape[0]
                loc_loss += loss.item()
                loc_steps += 1
                if step % args.print_freq == 0:
                    print(
                        "{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                            step,
                            len(dataset) // args.batch_size,
                            loc_loss / loc_steps,
                            time() - st_time,
                        )
                    )
        print(
            "Total \t L: {:.3f} \t -- {:.3f}".format(
                loc_loss / loc_steps,
                time() - st_time,
            )
        )


    def train_qa_s2s(self, qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args):
        s2s_optimizer = AdamW(qa_s2s_model.parameters(), lr=s2s_args.learning_rate, eps=1e-8)
        s2s_scheduler = get_linear_schedule_with_warmup(
            s2s_optimizer,
            num_warmup_steps=400,
            num_training_steps=(s2s_args.num_epochs + 1) * math.ceil(len(s2s_train_dset) / s2s_args.batch_size),
        )
        for e in range(s2s_args.num_epochs):
            self.train_qa_s2s_epoch(
                qa_s2s_model,
                s2s_train_dset,
                qa_s2s_tokenizer,
                s2s_optimizer,
                s2s_scheduler,
                s2s_args,
                e,
                curriculum=(e == 0),
            )
            m_save_dict = {
                "model": qa_s2s_model.state_dict(),
                "optimizer": s2s_optimizer.state_dict(),
                "scheduler": s2s_scheduler.state_dict(),
            }
            print("Saving model {}".format(s2s_args.model_save_name))
            self.eval_qa_s2s_epoch(qa_s2s_model, s2s_valid_dset, qa_s2s_tokenizer, s2s_args)
            torch.save(m_save_dict, "{}_{}.pth".format(s2s_args.model_save_name, e))