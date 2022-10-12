from typing import Sequence, Type, Union

import torch.nn as nn
import torch

from grad_example_module import quantize_int8

sp_ratio = 0.3
def batch_sparse(m, ratio=sp_ratio):

    m2 = m.reshape(m.shape[0], -1)
    v = torch.abs(m2)

    topk = torch.topk(v, max(1, int(v.shape[1]*ratio)))

    m2[v < topk.values[:, -1:]] = 0

    m3 = m2.reshape(m.shape)

    return m3

def naive_sparse(m, ratio):
    m2 = m.reshape(-1)
    v = torch.abs(m2)

    topk = torch.topk(v, max(1, int(v.shape[-1]*ratio)))

    m2[v <= topk.values[-1]] = 0

    m3 = m2.reshape(m.shape)

    return m3  

def batch_structural_sparse(m, ratio):
    # For N x T x H structure, it zeros some H.
    m2 = (m ** 2).sum(1)
    for i in range(m.shape[0]):
        m[i, :,  m2.topk(min(m.shape[-1] - 1, int(m.shape[-1] * (1 - ratio))), 1, largest=False, sorted=True).indices[i,:]] = 0
    return m

quant_bit = 8
def batch_quantization(m, bit=8):
    max_int = 2**(bit-1)
    m_shape = m.shape
    m = m.reshape(m.shape[0], -1)
    max_m = torch.abs(m).max(dim=1, keepdim=True)
    m = (m*max_int)/(max_m.values)
    m = m.to(torch.int32)
    m = m.to(torch.float32)
    m = (m*max_m.values)/max_int
    m = m.reshape(m_shape)
    return m

def naive_quantization(m, bit=8):
    max_int = 2**(bit-1)
    m_shape = m.shape
    m = m.reshape(-1)
    max_m = torch.abs(m).max(dim=1, keepdim=True)
    # print(1, m.max(), m.min())
    m = (m*max_int)/(max_m.values)
    # m = torch.floor(m)
    m = m.to(torch.int32)
    m = m.to(torch.float32)
    m = (m*max_m.values)/max_int
    m = m.reshape(m_shape)
    return m

def stochastic_rounding(x: torch.Tensor):
    floor = torch.floor(x)
    plus_one = (torch.rand_like(x) < (x - floor)).to(torch.float)
    return floor + plus_one

encode_stream = torch.cuda.Stream()

def batch_quantization_encode(m, bit=8):
    torch.cuda.set_stream(encode_stream)
    # max_int = 2**(bit-1)
    # m_shape = m.shape
    # m = m.reshape(m.shape[0], -1)
    # max_m = torch.abs(m).max()#dim=1, keepdim=True)
    # m = m*(max_int/max_m) #.values)
    # m = torch.round(m)
    # m = m.to(torch.int)
    # m = stochastic_rounding(m)
    # m = m.to(torch.int8)
    # m = m.to(torch.float32)
    # m = m.reshape(m_shape)
    m, max_m = quantize_int8(m)
    torch.cuda.set_stream(torch.cuda.default_stream())
    return m, max_m #.values

def batch_quantization_decode(m, max_m, bit=8):
    max_int = 2**(bit-1)
    # m_shape = m.shape
    # m = m.reshape(m.shape[0], -1)
    m = m.to(torch.float32)
    m = (m*max_m)/max_int
    # m = m.reshape(m_shape)
    return m

int_multiply = True
def compression(x):
    return x