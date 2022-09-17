
import pynvml
#from wandb.vendor.pynvml import pynvml as nvml
import torch

from dataclasses import dataclass

@dataclass
class GpuInfo:
    memory: int
    gpu: int

def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        first_gpu = torch.cuda.get_device_name(0)

        print(f'There are {n_gpus} GPU(s) available.')
        print(f'GPU gonna be used: {first_gpu}')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        n_gpus = 0
    return device, n_gpus

def customNvmlDeviceGetUtilizationRates(_arg):
    #_device, n_gpus = get_device()
    device = torch.cuda.current_device()
    return GpuInfo(gpu=device, memory=40960)

def setup():
    try:
        pynvml.nvmlInit()
        pynvml.nvmlDeviceGetUtilizationRates = customNvmlDeviceGetUtilizationRates
        #nvml.nvmlInit()
        #nvml.nvmlDeviceGetUtilizationRates = customNvmlDeviceGetUtilizationRates
        print("Initalized correctly globally")
    except:
        print("Failure to initalize globally")
