#############################
#   Imports
#############################

# Python modules

# Remote modules

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

def clean_mask_labels(label):
    return label.replace("[MASK]", "<mask>")