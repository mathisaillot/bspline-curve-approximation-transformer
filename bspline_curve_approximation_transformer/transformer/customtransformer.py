import torch
import math

import einops
from torch import Tensor
from torch import nn, einsum
from einops import rearrange, reduce


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


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            residual=True,
            residual_conv_kernel=33,
            dropout=0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.heads = heads
        project_out = not (heads == 1 and dim_head == dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        # dropout
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.residual = residual
        # TODO fix this residual by default
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None):

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        mask_value = max_neg_value(dots)

        input_mask = mask

        if exists(input_mask):
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~input_mask, mask_value)
            del input_mask

        attn = self.attend(dots)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        if self.residual:
            out += self.res_conv(v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        out = self.to_out(out)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out


class FullTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


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


class CustomTransformer(nn.Module):
    def __init__(self,
                 hidden_size: int = 512,
                 dim_head: int = 128,
                 depth: int = 6,
                 n_heads: int = 8,
                 dropout_pos_enc: float = 0.2,
                 dropout: float = 0.,
                 max_seq_len=2048,
                 n_class: list = [2],
                 input_depth: int = 1,
                 network_type: str = 'regression',
                 softmax: bool = False,
                 learnable_pe: bool = False,
                 no_pe: bool = False,
                 ):

        super().__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Linear(input_depth, hidden_size).float()
        self.network_type = network_type

        self.seq_len = max_seq_len + 1 if network_type == 'regression' or network_type == 'clstoken' else max_seq_len
        # Create positional encoder
        self.no_pe = no_pe
        if not no_pe:
            peclass = PositionalEncoderLearnable if learnable_pe else PositionalEncoder
            self.positional_encoding_layer = peclass(
                d_model=hidden_size,
                dropout=dropout_pos_enc,
                max_seq_len=self.seq_len
            )

        self.encoder = FullTransformer(dim=hidden_size,
                                       dim_head=dim_head,
                                       heads=n_heads,
                                       depth=depth,
                                       attn_dropout=dropout)

        self.single_output = len(n_class) == 1

        if self.single_output:
            self.classification_head = nn.Linear(hidden_size, n_class[0]).float()
        else:
            self.classification_heads = nn.ModuleList([nn.Linear(hidden_size, i).float() for i in n_class])

        if network_type == 'clstoken':
            self.norm = nn.LayerNorm(hidden_size)
            self.clstoken = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.func_out = nn.Softmax(dim=-1) if softmax else torch.tanh
        self.softmax = softmax
        self.n_class = n_class
        self.depth = depth

    def forward(self, src: Tensor, src_mask: Tensor = None, output_mask: list = None):
        """
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the feature number
            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence
            output_mask:
        """
        input_mask_expanded = src_mask.unsqueeze(-1).expand(src.size()).float()
        src = src * input_mask_expanded

        if self.network_type == 'regression':
            src = torch.cat((src, torch.ones_like(src[:, 0:1, :])), dim=1)
            src_mask = torch.cat((src_mask, torch.ones_like(src_mask[:, 0:1])), dim=1)

        src = self.embed(src)

        if self.network_type == 'clstoken':
            cls_tokens = self.clstoken.expand(src.shape[0], -1, -1)
            src = torch.cat((src, cls_tokens), dim=1)
            src_mask = torch.cat((src_mask, torch.ones_like(src_mask[:, 0:1])), dim=1)

        # Pass through the positional encoding layer
        if not self.no_pe:
            src = self.positional_encoding_layer(src)

        src = self.encoder(
            src,
            mask=src_mask.bool()
        )

        if self.network_type == 'clstoken':
            src = self.norm(src)

        if self.network_type == 'classic':
            input_mask_expanded = src_mask.unsqueeze(-1).expand(src.size()).float()
            sum_embeddings = torch.sum(src * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            if self.single_output:
                output = [self.classification_head(mean_embeddings)]
            else:
                output = [f(mean_embeddings) for f in self.classification_heads]
        else:
            out_net = src[:, -1:, :]
            if self.single_output:
                output = [self.classification_head(out_net).squeeze(-2)]
            else:
                output = [f(out_net).squeeze(-2) for f in self.classification_heads]

        if self.softmax:
            for i in range(len(output)):
                output[i] = output[i].masked_fill(~output_mask[i].bool(), -torch.finfo(output[i].dtype).max)

        return [self.func_out(i) for i in output]
