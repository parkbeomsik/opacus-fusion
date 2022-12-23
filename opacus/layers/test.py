import torch
# from torch.nn import LSTM
from opacus.layers.dp_rnn import DPLSTM
from opacus.layers.dp_fast_rnn import DPFASTLSTM



torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)

lstm = DPLSTM(input_size=4,
            hidden_size=1,
            num_layers=1,
            batch_first=True).cuda()

def print_grads(module, grad_input, grad_output):
    print("grads", grad_output)

lstm.cells[0].ih.register_full_backward_hook(print_grads)
lstm.cells[0].hh.register_full_backward_hook(print_grads)

dp_lstm = DPFASTLSTM(input_size=4,
                     hidden_size=1,
                     batch_first=True).cuda()

with torch.no_grad():
    dp_lstm.weight_ih[:] = lstm.weight_ih_l0[:]
    dp_lstm.weight_hh[:] = lstm.weight_hh_l0[:]

    lstm.bias_hh_l0.zero_()
    
    dp_lstm.bias[:] = lstm.bias_ih_l0[:]

input_data = torch.rand((3, 2, 4)).cuda().requires_grad_(True)
input_data2 = input_data.detach().clone().requires_grad_(True)

print(input_data)

lstm_out = lstm(input_data)
dp_lstm_out = dp_lstm(input_data2)

print(lstm_out[0])
print(dp_lstm_out[0])

output_grad = torch.rand((3, 2, 1)).cuda()
output_grad2 = output_grad.detach().clone()
lstm_out[0].backward(output_grad)
dp_lstm_out[0].backward(output_grad2)

print(input_data.grad)
print(input_data2.grad)

print(lstm.weight_ih_l0.grad)
print(dp_lstm.weight_ih.grad)

print(lstm.weight_hh_l0.grad)
print(dp_lstm.weight_hh.grad)