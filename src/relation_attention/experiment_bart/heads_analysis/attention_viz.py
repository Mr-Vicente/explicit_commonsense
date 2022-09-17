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
from torch import Tensor
from tqdm import tqdm

import wandb
from scipy.stats import t

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
    def __init__(self, device):
        self.device = device

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
        #plt.show()
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

    def plot_attn_lines(self, data, heads):
        """Plots attention maps for the given example and attention heads."""
        width = 3
        example_sep = 3
        word_height = 1
        pad = 0.1

        for ei, (layer, head) in enumerate(heads):
            yoffset = 1
            xoffset = ei * width * example_sep

            attn = data["attns"][layer][head]
            attn = np.array(attn)
            attn /= attn.sum(axis=-1, keepdims=True)
            words = data["tokens"]
            words[0] = "..."
            n_words = len(words)

            for position, word in enumerate(words):
                plt.text(xoffset + 0, yoffset - position * word_height, word,
                         ha="right", va="center")
                plt.text(xoffset + width, yoffset - position * word_height, word,
                         ha="left", va="center")
            for i in range(1, n_words):
                for j in range(1, n_words):
                    plt.plot([xoffset + pad, xoffset + width - pad],
                             [yoffset - word_height * i, yoffset - word_height * j],
                             color="blue", linewidth=1, alpha=attn[i, j])

    def plot_attn_lines_concepts(self, title, examples, layer, head, color_words,
                  color_from=True, width=3, example_sep=3,
                  word_height=1, pad=0.1, hide_sep=False):
        # examples -> {'words': tokens, 'attentions': [layer][head]}
        plt.figure(figsize=(4, 4))
        for i, example in enumerate(examples):
            yoffset = 0
            if i == 0:
                yoffset += (len(examples[0]["words"]) -
                            len(examples[1]["words"])) * word_height / 2
            xoffset = i * width * example_sep
            attn = example["attentions"][layer][head]
            if hide_sep:
                attn = np.array(attn)
                attn[:, 0] = 0
                attn[:, -1] = 0
                attn /= attn.sum(axis=-1, keepdims=True)

            words = example["words"]
            n_words = len(words)
            for position, word in enumerate(words):
                for x, from_word in [(xoffset, True), (xoffset + width, False)]:
                    color = "k"
                    if from_word == color_from and word in color_words:
                        color = "#cc0000"
                    plt.text(x, yoffset - (position * word_height), word,
                             ha="right" if from_word else "left", va="center",
                             color=color)

            for i in range(n_words):
                for j in range(n_words):
                    color = "b"
                    if words[i if color_from else j] in color_words:
                        color = "r"
                    print(attn[i, j])
                    plt.plot([xoffset + pad, xoffset + width - pad],
                             [yoffset - word_height * i, yoffset - word_height * j],
                             color=color, linewidth=1, alpha=attn[i, j])
        plt.axis("off")
        plt.title(title)
        plt.show()

    def plot_attn_lines_concepts_ids(title, examples, layer, head, color_words,
                                     relations_total,
                                     color_from=True, width=3, example_sep=3,
                                     word_height=1, pad=0.1, hide_sep=False):
        # examples -> {'words': tokens, 'attentions': [layer][head]}
        plt.clf()
        plt.figure(figsize=(10, 5))
        # print('relations_total:', relations_total)
        # print(examples[0])
        for idx, example in enumerate(examples):
            yoffset = 0
            if idx == 0:
                yoffset += (len(examples[0]["words"]) -
                            len(examples[0]["words"])) * word_height / 2
            xoffset = idx * width * example_sep
            attn = example["attentions"][layer][head]
            if hide_sep:
                attn = np.array(attn)
                attn[:, 0] = 0
                attn[:, -1] = 0
                attn /= attn.sum(axis=-1, keepdims=True)

            words = example["words"]
            n_words = len(words)
            example_rel = relations_total[idx]
            for position, word in enumerate(words):
                for x, from_word in [(xoffset, True), (xoffset + width, False)]:
                    color = "k"
                    for y_idx, y in enumerate(words):
                        if from_word and example_rel[position, y_idx] > 0:
                            # print('outgoing', position, y_idx)
                            color = "r"
                        if not from_word and example_rel[y_idx, position] > 0:
                            # print('coming', position, y_idx)
                            color = "g"
                    # if from_word == color_from and word in color_words:
                    #    color = "#cc0000"
                    plt.text(x, yoffset - (position * word_height), word,
                             ha="right" if from_word else "left", va="center",
                             color=color)

            for i in range(n_words):
                for j in range(n_words):
                    color = "k"
                    # print(i,j, example_rel[i,j])
                    if example_rel[i, j].item() > 0 and i <= j:
                        color = "r"
                    if example_rel[i, j].item() > 0 and i >= j:
                        color = "g"
                    plt.plot([xoffset + pad, xoffset + width - pad],
                             [yoffset - word_height * i, yoffset - word_height * j],
                             color=color, linewidth=1, alpha=attn[i, j])
                    # color=color, linewidth=1, alpha=min(attn[i, j]*10,1))
        plt.axis("off")
        plt.title(title)
        plt.show()


    def relation_binary_2d_to_1d(self, relations_binary_mask):
        relations_binary_mask = relations_binary_mask.sum(dim=1)
        relations_binary_mask[relations_binary_mask > 1] = 1
        return relations_binary_mask

    def compute_confidence_interval(self, data: Tensor,
                                          confidence: float = 0.95
                                          ):
        """
        Computes the confidence interval for a given survey of a data set.
        """
        n = len(data)
        mean: Tensor = data.mean()
        # se: Tensor = scipy.stats.sem(data)  # compute standard error
        # se, mean: Tensor = torch.std_mean(data, unbiased=True)  # compute standard error
        se: Tensor = data.std(unbiased=True) / (n ** 0.5)
        t_p: float = float(t.ppf((1 + confidence) / 2., n - 1))
        ci = t_p * se
        return mean, ci

    def get_head_mask(self, mat):
        mean, ci = self.compute_confidence_interval(mat)
        print('mat:', mat)
        print('mean:', mean)
        print('ci:', ci)
        head_mask = (mat < mean - ci) & (mat > mean + ci)
        #head_mask = (mat < mean + ci) & (mat > mean - ci)
        return head_mask.to(self.device)

    def compute_heads_importance(self,
                                 args,
                                 model,
                                 dataloader,
                                 compute_entropy=True,
                                 compute_importance=True,
                                 head_mask=None,
                                 commonsense_head_analysis=False,
                                 ):

        # Prepare our tensors
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
        head_importance = torch.zeros(n_layers, n_heads).to(args.device)
        simple_head_importance = torch.zeros(n_layers, n_heads).to(args.device)
        attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)

        commonsense_capture = None
        attn_commonsense_entropy = None
        if commonsense_head_analysis:
            commonsense_capture = torch.zeros(n_layers, n_heads).to(args.device)
            attn_commonsense_entropy = torch.zeros(n_layers, n_heads).to(args.device)

        if head_mask is None:
            head_mask = torch.ones(n_layers, n_heads).to(args.device)

        # To store gradients
        head_mask.requires_grad_(requires_grad=True)

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
                    #print('input_commonsense_relations: ',input_commonsense_relations.shape)
                    #print('inputs["attention_mask"]: ', inputs["attention_mask"].shape)
                    #print('attn.detach().shape: ', attn.detach().shape)
                    if commonsense_head_analysis:
                        relations_binary_mask = input_commonsense_relations#.clone()
                        #relations_binary_mask[relations_binary_mask > 1] = 1
                        print('relations_binary_mask: ', relations_binary_mask)
                        print('relations_binary_mask.shape: ', relations_binary_mask.shape)

                        bsz, num_heads, src_len, tgt_len = attn.size()
                        attn_det = attn.detach()
                        norm_attn = normalize(attn_det)
                        diff = abs(relations_binary_mask.view(bsz, 1, src_len, tgt_len) - norm_attn)
                        diff = diff.view(num_heads, bsz, src_len, tgt_len)
                        #commonsense_capture[layer] += torch.amax(diff, dim=(1, 2, 3))
                        score_sum = diff.sum(dim=(2, 3))
                        score_mean = score_sum.mean(dim=1) # along the batch find average of diff between commonsense and not
                        commonsense_capture[layer] += score_mean

                        relations_1d = self.relation_binary_2d_to_1d(relations_binary_mask)
                        masked_entropy = entropy(attn.detach()) * relations_1d.float().unsqueeze(1)
                        attn_commonsense_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

                    masked_entropy = entropy(attn.detach()) * inputs["attention_mask"].float().unsqueeze(1)
                    attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()
                    bsz, num_heads, src_len, tgt_len = attn.size()
                    simple_head_importance[layer] += attn.detach().view(num_heads, bsz, src_len, tgt_len).amax(dim=(1, 2, 3))

            if compute_importance:
                head_importance += head_mask.grad.abs().detach()

            # Also store our logits/labels if we want to compute metrics afterwards
            """
            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)
            """

            tot_tokens += inputs["attention_mask"].float().detach().sum().data
            wandb.log({"loss": loss, "attn_entropy": attn_entropy, "head_importance": head_importance})

        # Normalize
        attn_entropy /= tot_tokens
        attn_commonsense_entropy /= tot_tokens
        head_importance /= tot_tokens


        # Layerwise importance normalization
        if not args.dont_normalize_importance_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not args.dont_normalize_global_importance:
            head_importance = (head_importance - head_importance.min()) / (
                        head_importance.max() - head_importance.min())

        heads = self.get_head_mask(head_importance)
        print(heads)
        """
        logger.info("Attention entropies")
        print_2d_tensor(normalize_zero(attn_entropy))
        self.visualize_matrix(scores_mat=normalize_zero(attn_entropy).detach().cpu(), label_name='attn_entropy')

        logger.info("Head importance scores")
        print_2d_tensor(head_importance)
        self.visualize_matrix(scores_mat=head_importance.detach().cpu(), label_name='head_importance')

        logger.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=args.device
        )
        head_ranks = head_ranks.view_as(head_importance)
        print_2d_tensor(head_ranks)
        self.visualize_matrix(scores_mat=head_ranks.detach().cpu(), label_name='head_ranks')

        images = [wandb.Image("figs/attn_entropy.png", caption="attn_entropy"),
                  wandb.Image("figs/head_importance.png", caption="head_importance"),
                  wandb.Image("figs/head_ranks.png", caption="head_ranks"),
        ]

        if commonsense_head_analysis:
            logger.info("Attention commonsense entropy")
            print_2d_tensor(normalize_zero(attn_commonsense_entropy))
            self.visualize_matrix(scores_mat=normalize_zero(attn_commonsense_entropy).detach().cpu(), label_name='attn_commonsense_entropy')

            logger.info("Diff entropies")
            norm_diff_entropy = normalize_zero(normalize_zero(attn_entropy) - normalize_zero(attn_commonsense_entropy))
            print_2d_tensor(norm_diff_entropy)
            self.visualize_matrix(scores_mat=norm_diff_entropy.detach().cpu(), label_name='norm_diff_entropy')

            logger.info("Commonsense capture")
            commonsense_capture = normalize_zero(commonsense_capture)
            print_2d_tensor(commonsense_capture)
            self.visualize_matrix(scores_mat=commonsense_capture.detach().cpu(), label_name='commonsense_capture')

            images.append(wandb.Image("figs/commonsense_capture.png", caption="commonsense_capture"))
            images.append(wandb.Image("figs/attn_commonsense_entropy.png", caption="attn_commonsense_entropy"))
            images.append(wandb.Image("figs/norm_diff_entropy.png", caption="norm_diff_entropy"))

        wandb.log({"images": images})

        return attn_entropy, head_importance#, preds, labels
        """

def normalize(mat):
    new_mat = (mat - mat.min()) / (mat.max() - mat.min())
    return new_mat

def normalize_zero(mat):
    new_mat = mat/ mat.max()
    return new_mat