import math
import torch

# Our module!
import custom_rnn

from opacus.custom_tensor import GradOutputs

input_actvs = []
output_grads = []

class LSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input,
                weight_ih, weight_hh, bias,
                weight_ih_reverse, weight_hh_reverse, bias_reverse,
                old_h, old_cell, bidirectional=True):

        en_bias = bias is not None
        if bias is None:
            bias = torch.zeros_like(old_h)
        else:
            bias = bias.unsqueeze(0)
        if bidirectional:
            if bias_reverse is None:
                bias_reverse = torch.zeros_like(old_h)
            else:
                bias_reverse = bias_reverse.unsqueeze(0)

        if not bidirectional:
            weight_ih_reverse = torch.empty((0,))
            weight_hh_reverse = torch.empty((0,))
            bias_reverse = torch.empty((0,))

        output, new_h, new_cell, i_actv, \
        h_actv, c_actv, gate_actv, \
        h_actv_reverse, c_actv_reverse, gate_actv_reverse \
            = custom_rnn.lstm_forward(input,
                                      weight_ih, weight_hh, bias,
                                      weight_ih_reverse, weight_hh_reverse, bias_reverse,
                                      old_h, old_cell, bidirectional)

        if bidirectional:
            variables = [weight_ih, weight_hh,
                         weight_ih_reverse, weight_hh_reverse,
                         i_actv,
                         h_actv, c_actv, gate_actv,
                         h_actv_reverse, c_actv_reverse, gate_actv_reverse,
                         bias]
        else:
            variables = [weight_ih, weight_hh, i_actv, h_actv, c_actv, gate_actv, bias]
        ctx.save_for_backward(*variables)
        ctx.bidirectional = bidirectional
        ctx.en_bias = en_bias

        N = input.shape[0]
        L = input.shape[1]
        H = old_h.shape[2]
        
        new_i_actv = i_actv.detach() # [N, L, I]
        new_h_actv = h_actv.detach().transpose(0, 1) # [N, L, H]
        if bidirectional:
            new_h_actv_reverse = h_actv_reverse.detach().transpose(0, 1) # [N, L, H]

        global input_actvs
        if bidirectional:
            input_actvs = [new_i_actv, new_h_actv, new_h_actv_reverse]
        else:
            input_actvs = [new_i_actv, new_h_actv]

        return output, new_h, new_cell

    @staticmethod
    def backward(ctx, grad_output, grad_h, grad_cell):     

        bidirectional = ctx.bidirectional
        en_bias = ctx.en_bias

        if bidirectional:
            weight_ih, weight_hh,\
            weight_ih_reverse, weight_hh_reverse,\
            i_actv,\
            h_actv, c_actv, gate_actv,\
            h_actv_reverse, c_actv_reverse, gate_actv_reverse,\
            bias = ctx.saved_tensors
        else:
            weight_ih, weight_hh, i_actv, h_actv, c_actv, gate_actv, bias = ctx.saved_tensors
            weight_ih_reverse, weight_hh_reverse = torch.empty((0,)), torch.empty((0,))
            h_actv_reverse, c_actv_reverse, gate_actv_reverse = torch.empty((0,)), torch.empty((0,)), torch.empty((0,))

        outputs = custom_rnn.lstm_backward(
            grad_output, grad_h.contiguous(), grad_cell.contiguous(),
            weight_ih, weight_hh, weight_ih_reverse, weight_hh_reverse,
            i_actv,
            h_actv, c_actv, gate_actv,
            h_actv_reverse, c_actv_reverse, gate_actv_reverse,
            bidirectional)
        d_input, d_old_h, d_old_cell, grad_gates, grad_gates_reverse = outputs

        N = i_actv.shape[0]
        L = i_actv.shape[1]
        I = weight_ih.shape[1]
        H = d_old_h.shape[2]

        grad_weight_ih = None
        grad_weight_hh = None
        grad_weight_ih_reverse = None
        grad_weight_hh_reverse = None
        grad_bias = None
        grad_bias_reverse = None

        # print("DPFASTLSTM grad_gates", grad_gates)
        # print("DPFASTLSTM i_actv", i_actv)

        if weight_ih.requires_grad:
            grad_weight_ih = torch.matmul(grad_gates.reshape(L*N, 4*H).transpose(0, 1),
                                         i_actv.transpose(0, 1).reshape(L*N, I))
        if weight_hh.requires_grad:
            grad_weight_hh = torch.matmul(grad_gates.reshape(L*N, 4*H).transpose(0, 1),
                                         h_actv.reshape(L*N, H))
        if en_bias:
            if bias.requires_grad:
                grad_bias = grad_gates.sum(dim=(0, 1), keepdim=False)

        if bidirectional:
            if weight_ih.requires_grad:
                grad_weight_ih_reverse = torch.matmul(grad_gates_reverse.reshape(L*N, 4*H).transpose(0, 1),
                                            i_actv.transpose(0, 1).reshape(L*N, I))
            if weight_hh.requires_grad:
                grad_weight_hh_reverse = torch.matmul(grad_gates_reverse.reshape(L*N, 4*H).transpose(0, 1),
                                            h_actv_reverse.reshape(L*N, H))
            if en_bias:
                if bias.requires_grad:
                    grad_bias_reverse = grad_gates_reverse.sum(dim=(0, 1), keepdim=False)

        D = 2 if bidirectional else 1

        new_grad_gates = grad_gates.detach()
        new_grad_gates = new_grad_gates.reshape(L, N, 4*H).transpose(0, 1) # [N, L, 4*H]
        if bidirectional:
            new_grad_gates_reverse = grad_gates_reverse.detach()
            new_grad_gates_reverse = new_grad_gates_reverse.reshape(L, N, 4*H).transpose(0, 1)

        global output_grads
        if bidirectional:
            output_grads = [new_grad_gates, new_grad_gates_reverse]
        else:
            output_grads = [new_grad_gates]

        return d_input, grad_weight_ih, grad_weight_hh, grad_bias, \
            grad_weight_ih_reverse, grad_weight_hh_reverse, grad_bias_reverse, \
            d_old_h, d_old_cell, None


class DPFASTLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, batch_first=True, bidirectional=False):
        super(DPFASTLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(
            torch.empty(4 * hidden_size, input_size))
        torch.nn.init.uniform_(self.weight_ih.data, -0.1, 0.1)
        self.weight_hh = torch.nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size))
        torch.nn.init.uniform_(self.weight_hh.data, -0.1, 0.1)
        self.en_bias = False
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(4 * hidden_size))
            torch.nn.init.uniform_(self.bias.data, -0.1, 0.1)
            self.en_bias = True
        self.weight_ih_reverse = None
        self.weight_hh_reverse = None
        self.bias_reverse = None
        if bidirectional:
            self.weight_ih_reverse = torch.nn.Parameter(
                torch.empty(4 * hidden_size, input_size))
            torch.nn.init.uniform_(self.weight_ih.data, -0.1, 0.1)
            self.weight_hh_reverse = torch.nn.Parameter(
                torch.empty(4 * hidden_size, hidden_size))
            torch.nn.init.uniform_(self.weight_hh.data, -0.1, 0.1)
            if bias:
                self.bias_reverse = torch.nn.Parameter(torch.empty(4 * hidden_size))
                torch.nn.init.uniform_(self.bias_reverse.data, -0.1, 0.1)
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state=None):
        if state is not None:
            h_0, c_0 = state
        else:
            B = input.shape[0] if self.batch_first else input.shape[1]
            D = 2 if self.bidirectional else 1
            h_0 = torch.zeros(D, B, self.hidden_size).to(input.device)
            c_0 = torch.zeros(D, B, self.hidden_size).to(input.device)

        if not self.batch_first:
            input = input.transpose(0, 1) # [N, L, I]

        output = LSTMFunction.apply(input, 
                                    self.weight_ih, self.weight_hh, self.bias, 
                                    self.weight_ih_reverse, self.weight_hh_reverse, self.bias_reverse,
                                    h_0, c_0, self.bidirectional) 

        h_all = output[0]
        h_n = output[1]
        c_n = output[2]

        return h_all, (h_n, c_n)

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}, bias={en_bias}, batch_first={batch_first}, bidirectional={bidirectional}'
        return s.format(**self.__dict__)