import math
from tkinter.font import BOLD
from typing import List, Union
from opacus.layers.dp_fast_rnn import DPFASTLSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import CTCLoss


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=DPFASTLSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.group_norm = nn.GroupNorm(32, input_size) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True, batch_first=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        x = x.transpose(1, 2) # [N, H, L]
        if self.group_norm is not None:
            x = self.group_norm(x)
        x = x.transpose(1, 2) # [N, L, H]

        x, h = self.rnn(x)
        
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):
    def __init__(self,
                 bidirectional: bool,
                 labels: List,
                 num_hidden_layers: int = 5,
                 hidden_size: int = 2560):
        super().__init__()

        self.labels = labels
        num_classes = len(self.labels)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.GroupNorm(32, 32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.GroupNorm(32, 32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor(320 / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=hidden_size*(2 if bidirectional else 1),
                    hidden_size=hidden_size,
                    bidirectional=self.bidirectional
                ) for x in range(num_hidden_layers - 1)
            )
        )

        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(hidden_size, context=20),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        self.gn = nn.GroupNorm(32, hidden_size*(2 if bidirectional else 1))
        self.fc = nn.Linear(hidden_size*(2 if bidirectional else 1), num_classes, bias=False)
        
    def forward(self, x):
        lengths = torch.Tensor([x.shape[2] for _ in range(x.shape[0])])
        output_lengths = self.get_seq_lens(lengths)

        x = x.unsqueeze(1)
        x = self.conv(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2) # [N, L, I]

        for rnn in self.rnns:
            x = rnn(x) # [N, L, D*H]

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = x.transpose(0, 1) # [L, N, H]
            x = self.lookahead(x) # [L, N, H]
            x = x.transpose(0, 1) # [N, L, H]

        x = x.transpose(1, 2)     # [N, H, L]
        x = self.gn(x)
        x = x.transpose(1, 2)     # [N, L, H]
        x = self.fc(x)
        x = x.transpose(0, 1)     # [L, N, H]
        x = x.log_softmax(-1)
        
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()
