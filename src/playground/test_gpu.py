
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        first_gpu = torch.cuda.get_device_name(0)
        print(device)
        print(torch.cuda.get_arch_list())
        print(torch.cuda._get_device_index(device))

        print(f'There are {n_gpus} GPU(s) available.')
        print(f'GPU gonna be used: {first_gpu}')
