#############################
#   Imports
#############################

# Python modules

# Remote modules
import torch
from torch import Tensor
from scipy.stats import t

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class HeadsSupervisor:
    def __init__(self, n_layers:int, n_heads:int, device):
        self.heads_mask = torch.ones(n_layers, n_heads).to(device)
        self.heads_mask.requires_grad_(requires_grad=True)
        self.head_importance = torch.zeros(n_layers, n_heads).to(device)
        self.attn_entropy = torch.zeros(n_layers, n_heads).to(device)
        self.commonsense_capture = torch.zeros(n_layers, n_heads).to(device)
        self.device=device
        self.tot_tokens = 0
        self.updating=True
        self.curr_head_mask = None
        self.temp_data = {'encoder_attentions': []}

    def stop_updating(self):
        self.heads_mask.requires_grad_(requires_grad=False)
        self.updating=False

    def is_updating(self):
        return self.updating

    def reset_for_new_epoch(self):
        self.tot_tokens = 0
        self.head_importance = torch.zeros_like(self.head_importance)
        self.attn_entropy = torch.zeros_like(self.attn_entropy)

    def store_info_head_supervisors(self, examples_outputs):
        loss, logits, all_attentions, head_mask = (
            examples_outputs.loss,
            examples_outputs.logits,
            examples_outputs.encoder_attentions if 'encoder_attentions' in examples_outputs else [],
            examples_outputs.head_mask,
        )  # Loss and logits are the first, attention the last
        self.curr_head_mask = head_mask
        self.temp_data = {'encoder_attentions': all_attentions}

    def update_head_supervisors(self, inputs):
        all_attentions, head_mask = self.temp_data['encoder_attentions'], self.curr_head_mask
        #for layer, attn in enumerate(all_attentions):
        #    masked_entropy = self.calc_entropy(attn.detach()) * inputs["attention_mask"].float().unsqueeze(1)
        #    self.attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        curr_head_importance = head_mask.grad.abs().detach()
        #print('curr_head_importance:', curr_head_importance)
        curr_head_importance = curr_head_importance.masked_fill(torch.isnan(curr_head_importance), 0)
        self.head_importance += curr_head_importance
        #print('self.head_importance:', self.head_importance)
        curr_tokens = inputs["attention_mask"].float().detach().sum().data
        #print('curr_tokens:', curr_tokens)
        self.tot_tokens += curr_tokens

    def evaluate_heads(self):
        # Normalize
        #self.attn_entropy /= self.tot_tokens
        print('tot_tokens:', self.tot_tokens)
        print('meow head_importance:', self.head_importance)
        self.head_importance /= self.tot_tokens

        #att_entrop = self.attn_entropy.detach().cpu()
        head_imp = self.head_importance.detach().cpu()
        head_mask = self.get_head_mask(head_imp)
        print('self.heads_mask [before]:',  self.heads_mask)
        self.heads_mask = head_mask.float()
        print('self.heads_mask [after]:',  self.heads_mask)

    def get_head_mask(self, mat):
        mean, ci = self.compute_confidence_interval(mat)
        print('mat:', mat)
        print('mean:', mean)
        print('ci:', ci)
        head_mask = (mat < mean + ci) & (mat > mean - ci)
        #head_mask = (mat < mean + ci) & (mat > mean - ci)
        return head_mask.to(self.device)

    def heads_to_vec(self):
        return self.heads_mask.detach().cpu().tolist()

    #############################
    #   Helper function
    #############################

    def calc_entropy(self, p):
        """Compute the entropy of a probability distribution"""
        plogp = p * torch.log(p)
        plogp[p == 0] = 0
        return -plogp.sum(dim=-1)

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