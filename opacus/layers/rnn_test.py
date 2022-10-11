import torch
from dp_fast_rnn import DPFASTLSTM

input = torch.ones(32, 64, 1024).to(torch.device("cuda"))
i_weight = torch.ones(4096, 1024).to(torch.device("cuda"))
h_weight = torch.ones(4096, 1024).to(torch.device("cuda"))
bias = torch.ones(4096).to(torch.device("cuda"))
old_h = torch.ones(1, 32, 1024).to(torch.device("cuda"))
old_c = torch.ones(1, 32, 1024).to(torch.device("cuda"))

layer = DPFASTLSTM(1024, 1024, True, True, False).to(torch.device("cuda"))

output, state = layer(input, (old_h, old_c))

grad_output = torch.ones_like(output)

output.backward(grad_output)