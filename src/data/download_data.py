from collections import deque

from datasets import load_dataset
from utils import Data_Type, write_dict_2_json_file

datatype = Data_Type.COMMONGEN

if __name__ == '__main__':
    data_points = deque()
    if datatype == Data_Type.COMMONGEN:
        dataset = load_dataset("common_gen")
        #print(dataset['train'])
        #print(dataset['train'].data)
        data_split = ['train', 'validation']
        for d_split in data_split:
            data = dataset[d_split].data
            #print(data)
            for concepts, target in zip(data['concepts'], data['target']):
                #print(concepts, target)
                new_data_point = {
                    'input_data': ' '.join([str(x) for x in concepts]),
                    'labels': str(target)
                }
                #print(new_data_point)
                data_points.append(new_data_point)
        data_points = list(data_points)
        write_dict_2_json_file(data_points, 'commongen.json')