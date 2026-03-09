import math

import torch
from torch import nn, Tensor


def exists(val):
    return val is not None


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


"""
Also LearnableBoundedHardTanh
"""


class LearnedBoundedReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.ReLU()
        self.f2 = nn.ReLU()
        self.min_bound = nn.Parameter(1 * torch.ones(1))
        self.slope = nn.Parameter(1 * torch.ones(1))

    def forward(self, x):
        return self.f1(1 - self.f2(self.min_bound - (x * self.slope)))

class LearnedSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Sigmoid()
        self.min_bound = nn.Parameter(0.5 * torch.ones(1))
        self.slope = nn.Parameter(1 * torch.ones(1))

    def forward(self, x):
        return (self.f1(x) * self.slope) - self.min_bound


class BoundedReLU(nn.Module):
    def __init__(self, min_bound=1, slope=1):
        super().__init__()
        self.f1 = nn.ReLU()
        self.f2 = nn.ReLU()
        self.min_bound = min_bound
        self.slope = slope

    def forward(self, x):
        return self.f1(1 - self.f2(self.min_bound - (x * self.slope)))


class UnitInterval(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x > 1, 1, torch.where(x<0, 0, x))


def get_nn_activation(activation: str):
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim=-1)
    elif activation == "lbrelu":
        return LearnedBoundedReLU()
    elif activation == "lsigmoid":
        return LearnedSigmoid()
    elif activation == "brelu":
        return BoundedReLU()
    elif activation == "unit":
        return UnitInterval()

    return nn.Identity()

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class OutputLayer(nn.Module):
    def __init__(self, dim_in, dim_out, func_out):
        super().__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.func_out = get_nn_activation(func_out)

    def forward(self, x, mask=None, input_len=None):
        return self.func_out(self.layer(x))


class PositionalEncoder(nn.Module):
    """
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/utils.py
    """

    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512):
        """
        Args:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values
        # dependent on position and i
        position = torch.arange(max_seq_len).unsqueeze(1)
        exp_input = torch.arange(0, d_model, 2) * \
                    (-math.log(10000.0) / d_model)
        # Returns a new tensor with the exponential of the elements of exp_input
        div_term = torch.exp(exp_input)
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)

        # torch.Size([target_seq_len, dim_val])
        pe[:, 1::2] = torch.cos(position * div_term)

        # torch.Size([target_seq_len, input_size, dim_val])
        pe = pe.unsqueeze(0).transpose(0, 1)

        # register that pe is not a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """
        add = self.pe[:x.size(1), :].squeeze(1)
        x = x + add

        return self.dropout(x)


class PositionalEncoderLearnable(nn.Module):
    """
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/utils.py
    """

    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512):
        """
        Args:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.randn(max_seq_len, d_model) - 0.5)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """
        add = self.pe[:x.size(1), :].squeeze(1)
        x = x + add

        return self.dropout(x)
