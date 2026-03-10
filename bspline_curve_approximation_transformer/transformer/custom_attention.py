import torch
from einops import rearrange
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from bspline_curve_approximation_transformer.transformer import (exists, PreNorm, FeedForward)

from bspline_curve_approximation_transformer.utils import triton_util

if triton_util.TRITON_AVAILABLE:
    from bspline_curve_approximation_transformer.transformer import FlashAttention


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            residual=False,
            residual_conv_kernel=33,
            dropout=0.,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.heads = heads
        project_out = not (heads == 1 and dim_head == dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        # dropout
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.residual = residual
        self.triton = triton_util.TRITON_AVAILABLE and not residual

        # TODO fix this residual by default
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, input_len=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if self.triton:
            padding = input_len
            if padding is None and mask is not None:
                padding = torch.count_nonzero(mask, dim=1)
            # pdb.set_trace()
            out_test = FlashAttention(q, k, v, self.scale, padding)
        else:

            if self.residual:
                res = self.res_conv(v)
            input_mask = mask

            if input_mask is not None:
                input_mask = rearrange(input_mask, 'b j -> b 1 1 j')

            out_test = scaled_dot_product_attention(query=q, key=k, value=v, scale=self.scale,
                                                    is_causal=False,
                                                    dropout_p=self.dropout_p if self.training else 0.0,
                                                    attn_mask=input_mask)
            if self.residual:
                out_test = out_test + res

        out_test = rearrange(out_test, 'b h n d -> b n (h d)', h=self.heads)

        out_test = self.to_out(out_test)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out_test = out_test.masked_fill(~mask, 0.)

        return out_test


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim_q,
            dim_k,
            dim_head=64,
            heads=8,
            dropout=0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.heads = heads
        project_out = not (heads == 1 and dim_head == dim_k)

        self.to_q = nn.Linear(dim_q, inner_dim, bias=False)
        self.to_k = nn.Linear(dim_k, inner_dim, bias=False)
        self.to_v = nn.Linear(dim_k, inner_dim, bias=False)

        self.fn = nn.Softmax(dim=-1)

        # dropout
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_q),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y, mask=None):

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(y), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(y), 'b n (h d) -> b h n d', h=self.heads)

        input_mask = mask

        if input_mask is not None:
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')

        out_test = scaled_dot_product_attention(query=q, key=k, value=v, scale=self.scale,
                                                is_causal=False,
                                                dropout_p=self.dropout_p if self.training else 0.0,
                                                attn_mask=input_mask)

        out_test = rearrange(out_test, 'b h n d -> b n (h d)', h=self.heads)

        out_test = self.to_out(out_test)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out_test = out_test.masked_fill(~mask, 0.)

        return out_test


class AttentionHeads(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_head=64,
                 heads=8,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 residual=False,
                 ):
        super().__init__()
        self.attn = PreNorm(dim_in,
                            Attention(dim=dim_in, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                                      residual=residual))
        self.ff = PreNorm(dim_in, FeedForward(dim=dim_in, dropout=ff_dropout))

    def forward(self, x, mask=None, input_len=None):
        x = self.attn(x, mask=mask, input_len=input_len) + x
        return self.ff(x) + x


class CrossAttentionHeads(nn.Module):
    def __init__(self,
                 dim_x,
                 dim_y,
                 dim_head=64,
                 heads=8,
                 attn_dropout=0.,
                 ff_dropout=0.
                 ):
        super().__init__()
        self.attn = PreNorm(dim_x,
                            CrossAttention(dim_q=dim_x, dim_k=dim_y, dim_head=dim_head,
                                           heads=heads, dropout=attn_dropout))
        self.ff = PreNorm(dim_x, FeedForward(dim=dim_x, dropout=ff_dropout))

    def forward(self, x, y, mask=None):
        x = self.attn(x, y=y, mask=mask) + x
        return self.ff(x) + x
