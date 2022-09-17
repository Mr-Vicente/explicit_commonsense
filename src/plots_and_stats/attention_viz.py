#############################
#   Imports
#############################

# Python modules
import logging
import os
from datetime import datetime

# Remote modules
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Local modules

#############################
#   Constants
#############################

logger = logging.getLogger(__name__)

def entropy(p):
    """Compute the entropy of a probability distribution"""
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def print_2d_tensor(tensor):
    """Print a 2D tensor"""
    print("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor[0]))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))

class AttentionVisualizer:
    def __init__(self):
        pass

    def visualize_token2token_scores(self, all_tokens,
                                     scores_mat,
                                     useful_indeces,
                                     x_label_name='Head',
                                     apply_normalization=True):
        fig = plt.figure(figsize=(20, 20))

        all_tokens = np.array(all_tokens)[useful_indeces]
        for idx, scores in enumerate(scores_mat):
            if apply_normalization:
                scores = torch.from_numpy(scores)
                shape = scores.shape
                scores = scores.reshape((shape[0],shape[1], 1))
                scores = torch.linalg.norm(scores, dim=2)
            scores_np = np.array(scores)
            scores_np = scores_np[useful_indeces, :]
            scores_np = scores_np[:, useful_indeces]
            ax = fig.add_subplot(4, 4, idx + 1)
            # append the attention weights
            im = ax.imshow(scores_np, cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(all_tokens)))
            ax.set_yticks(range(len(all_tokens)))

            ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(all_tokens, fontdict=fontdict)
            ax.set_xlabel('{} {}'.format(x_label_name, idx + 1))

            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def visualize_matrix(self,
                         scores_mat,
                         label_name='heads_layers'):
        _fig = plt.figure(figsize=(20, 20))
        scores_np = np.array(scores_mat)
        fig, ax = plt.subplots()
        im = ax.imshow(scores_np, cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(scores_mat[0])))
        ax.set_yticks(range(len(scores_mat)))

        x_labels = [f'head-{i}' for i in range(1, len(scores_mat[0])+1)]
        y_labels = [f'layer-{i}' for i in range(1, len(scores_mat) + 1)]

        ax.set_xticklabels(x_labels, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(y_labels, fontdict=fontdict)
        ax.set_xlabel('{}'.format(label_name))

        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'figs/{label_name}.png', dpi=fig.dpi)

    def visualize_token2head_scores(self, all_tokens, scores_mat):
        fig = plt.figure(figsize=(30, 50))
        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(6, 3, idx + 1)
            # append the attention weights
            im = ax.matshow(scores_np, cmap='viridis')

            fontdict = {'fontsize': 20}

            ax.set_xticks(range(len(all_tokens)))
            ax.set_yticks(range(len(scores)))

            ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
            ax.set_xlabel('Layer {}'.format(idx + 1))

            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def relation_binary_2d_to_1d(self, relations_binary_mask):
        relations_binary_mask = relations_binary_mask.sum(dim=1)
        relations_binary_mask[relations_binary_mask > 1] = 1
        return relations_binary_mask

    def compute_heads_importance(self,
                                 args,
                                 model,
                                 dataloader,
                                 compute_entropy=True,
                                 compute_importance=True,
                                 head_mask=None,
                                 commonsense_head_analysis=False,
                                 actually_pruned=False):

        # Prepare our tensors
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
        head_importance = torch.zeros(n_layers, n_heads).to(args.device)
        attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)

        if head_mask is None:
            head_mask = torch.ones(n_layers, n_heads).to(args.device)

        # To store gradients
        head_mask.requires_grad_(requires_grad=True)

        preds = None
        labels = None
        tot_tokens = 0.0

        for step, inputs in enumerate(tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
            for k, v in inputs.items():
                inputs[k] = v.to(args.device)

            # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
            input_commonsense_relations = inputs.get("input_commonsense_relations")
            if 'input_commonsense_relations' in inputs:
                input_commonsense_relations = input_commonsense_relations.clone()
                input_commonsense_relations[input_commonsense_relations > 1] = 1
                inputs.pop('input_commonsense_relations')
            outputs = model(**inputs, head_mask=head_mask, output_attentions=True)
            loss, logits, all_attentions = (
                outputs.loss,
                outputs.logits,
                outputs.encoder_attentions,
            )  # Loss and logits are the first, attention the last
            loss.backward()  # Backpropagate to populate the gradients in the head mask

            if compute_entropy:
                for layer, attn in enumerate(all_attentions):
                    print('input_commonsense_relations: ',input_commonsense_relations.shape)
                    print('inputs["attention_mask"]: ', inputs["attention_mask"].shape)
                    print('attn.detach().shape: ', attn.detach().shape)
                    if commonsense_head_analysis:
                        """
                        input_commonsense_relations = input_commonsense_relations.float().unsqueeze(0)
                        print('in: ', input_commonsense_relations.size())
                        at = attn.detach()
                        bsz, num_heads, src_len, tgt_len = at.size()
                        print('at: ', bsz, num_heads, src_len, tgt_len)
                        at = at.reshape(num_heads, bsz, src_len, tgt_len)
                        print('s:', entropy(at).shape)
                        masked_entropy = entropy(at) * input_commonsense_relations
                        
                        masked_entropy = masked_entropy.reshape(bsz, num_heads, src_len, tgt_len)
                        """
                        relations_binary_mask = input_commonsense_relations.clone()
                        relations_binary_mask[relations_binary_mask > 1] = 1
                        relations_1d = self.relation_binary_2d_to_1d(relations_binary_mask)
                        masked_entropy = entropy(attn.detach()) * relations_1d.float().unsqueeze(1)
                    else:
                        masked_entropy = entropy(attn.detach()) * inputs["attention_mask"].float().unsqueeze(1)
                    attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

            if compute_importance:
                head_importance += head_mask.grad.abs().detach()

            # Also store our logits/labels if we want to compute metrics afterwards
            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

            tot_tokens += inputs["attention_mask"].float().detach().sum().data

        # Normalize
        attn_entropy /= tot_tokens
        head_importance /= tot_tokens

        # Layerwise importance normalization
        if not args.dont_normalize_importance_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not args.dont_normalize_global_importance:
            head_importance = (head_importance - head_importance.min()) / (
                        head_importance.max() - head_importance.min())

        logger.info("Attention entropies")
        print_2d_tensor(attn_entropy)
        self.visualize_matrix(scores_mat=attn_entropy, label_name='attn_entropy')
        logger.info("Head importance scores")
        print_2d_tensor(head_importance)
        self.visualize_matrix(scores_mat=head_importance, label_name='head_importance')
        logger.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=args.device
        )
        head_ranks = head_ranks.view_as(head_importance)
        print_2d_tensor(head_ranks)
        self.visualize_matrix(scores_mat=head_ranks, label_name='head_ranks')

        return attn_entropy, head_importance, preds, labels