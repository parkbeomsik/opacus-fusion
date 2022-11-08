import torch
from grad_example_module import quantize_int8

t = torch.rand((32, 32, 768)).to(torch.device("cuda"))

m, max_m = quantize_int8(t)
