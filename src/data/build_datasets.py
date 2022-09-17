#############################
#   Imports
#############################

# Python modules

# Remote modules

# Local modules
from utils import Data_Type
from cleansing_utils import Data_Cleaner

#############################
#   Constants
#############################

#DATASETS = [Data_Type.ELI5, Data_Type.STACK_EXCHANGE]
DATASETS = [Data_Type.COMMONSENSE_QA]

if __name__ == '__main__':
    data_builder_cleaner = Data_Cleaner()
    data_builder_cleaner.build_and_clean_datasets(DATASETS)
