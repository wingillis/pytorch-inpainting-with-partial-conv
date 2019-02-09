import torch
import opt


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor([1, 1, 1]) + torch.Tensor([1, 1, 1])
    x = x.transpose(1, 3)
    return x
