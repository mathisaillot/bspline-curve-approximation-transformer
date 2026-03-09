import torch
from torch import Tensor
from torch import nn

from bspline_curve_approximation_transformer.transformer import (Attention, FeedForward, PreNorm,
                                                                 PositionalEncoder, PositionalEncoderLearnable,
                                                                 get_nn_activation)


class FullTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            residual=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                                  residual=residual)),
                PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout))
            ]))

    def forward(self, x, mask=None,  input_len=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, input_len=input_len) + x
            x = ff(x) + x
        return x


class CustomTransformer(nn.Module):
    def __init__(self,
                 hidden_size: int = 512,
                 dim_head: int = 128,
                 depth: int = 6,
                 n_heads: int = 8,
                 dropout_pos_enc: float = 0.2,
                 attn_dropout: float = 0.,
                 ff_dropout: float = 0.,
                 max_seq_len=2048,
                 n_class: list = [2],
                 input_depth: int = 1,
                 network_type: str = 'regression',
                 softmax: bool = False,
                 learnable_pe: bool = False,
                 no_pe: bool = False,
                 residual: bool = False,
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
                                       attn_dropout=attn_dropout,
                                       ff_dropout=ff_dropout,
                                       residual=residual)

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

    def forward(self, src: Tensor, src_mask: Tensor = None, output_mask: list = None, input_len: Tensor = None):
        input_mask_expanded = src_mask.unsqueeze(-1).float()
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
            mask=src_mask.bool(),
            input_len=input_len
        )

        if self.network_type == 'clstoken':
            src = self.norm(src)

        if self.network_type == 'classic':
            input_mask_exp = src_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(src * input_mask_exp, 1)
            sum_mask = src_mask.sum(1)
            # sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask.unsqueeze(1)
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


class UMAPTransformer(nn.Module):
    def __init__(self,
                 hidden_size: int = 512,
                 dim_head: int = 128,
                 depth: int = 6,
                 n_heads: int = 8,
                 dropout_pos_enc: float = 0.2,
                 attn_dropout: float = 0.,
                 ff_dropout: float = 0.,
                 max_seq_len=2048,
                 output_depth: tuple | list = (1,),
                 input_depth: int = 1,
                 func_out: str = 'ReLU',
                 learnable_pe: bool = False,
                 no_pe: bool = False,
                 residual: bool = False,
                 ):

        super().__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Linear(input_depth, hidden_size).float()

        self.seq_len = max_seq_len
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
                                       attn_dropout=attn_dropout,
                                       ff_dropout=ff_dropout,
                                       residual=residual)

        self.classification_heads = nn.ModuleList([nn.Linear(hidden_size, i).float() for i in output_depth])

        self.func_out = get_nn_activation(func_out)

        self.output_depth = output_depth
        self.depth = depth

    def forward(self, src: Tensor, src_mask: Tensor = None,
                output_mask: Tensor = None, input_len: Tensor = None) -> list | Tensor:
        input_mask_expanded = src_mask.unsqueeze(-1).float()

        src = src * input_mask_expanded

        src = self.embed(src)

        # Pass through the positional encoding layer
        if not self.no_pe:
            src = self.positional_encoding_layer(src)

        src = self.encoder(
            src,
            mask=src_mask.bool(),
            input_len=input_len
        )

        output = [head(src) for head in self.classification_heads]

        if self.func_out is not None:
            output = [self.func_out(i) for i in output]

        return output
