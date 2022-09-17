
#############################
#   Imports
#############################

# Python modules

# Remote modules
from torch.utils.data import Dataset

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

class DefaultDataset(Dataset):
    def __init__(
            self,
            data,
            device=None,
            limitation=None,
    ):
        self.data = data[:limitation] if limitation else data
        self.device = device

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class RelationsDataset(Dataset):
    def __init__(
            self,
            data,
            device=None,
            max_length=128,
            limitation=None,
    ):
        self.data = data[:limitation] if limitation else data
        self.device = device
        self.max_length = max_length

    def __getitem__(self, idx):
        # d = {k:v.to(self.device) for k,v in self.data[idx].items()}
        #print(self.data[idx].get('input_commonsense_relations').sum())
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class MaskRelationsDataset(RelationsDataset):
    def __init__(self,
            data,
            device=None,
            max_length=128,
            limitation=None
    ):
        super(MaskRelationsDataset, self).__init__(
            data,
            device=device,
            max_length=max_length,
            limitation=limitation
        )
