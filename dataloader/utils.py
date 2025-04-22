import torch

def torch_random_int(low, high):
    return  torch.randint(low = low,
                          high = high,
                          size = (1,)).item()

