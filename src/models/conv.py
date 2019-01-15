""" A ripped version of fairseq convolutional seq2seq encoder """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single

from ..utils import make_positions

class FConvEncoder(nn.Module):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.
    Args:
        vocab (~fairseq.data.vocab): encoding vocab
        embed_dim (int, optional): embedding dimension
        embed_dict (str, optional): filename from which to load pre-trained
            embeddings
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
        normalization_constant (float, optional): multiplies the result of the
            residual block by sqrt(value)
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(
            self, vocab, input_dim=512, output_dim=512, max_positions=1024,
            convolutions=((512, 3),) * 20, dropout=0.1, left_pad=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.left_pad = left_pad
        #self.num_attention_layers = None
        self.input_dim = input_dim

        self.padding_idx = vocab.get_token_index(vocab._padding_token)
        self.embed_positions = PositionalEmbedding(max_positions, input_dim,
                                                   self.padding_idx, left_pad=self.left_pad)

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(input_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                Conv1d(in_channels, out_channels * 2, kernel_size, dropout=dropout, padding=padding)
                #ConvTBC(in_channels, out_channels * 3, kernel_size, dropout=dropout, padding=padding)
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = Linear(in_channels, output_dim)

    def forward(self, src_embs, src_masks):
        """
        Args:
            src_embs (FloatTensor): embeddings of shape `(batch, src_len, emb_dim)`
            src_masks (LongTensor): binary mask of each sentence of shape `(batch, src_len)`
        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, embed_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # we receive token embeddings; so we only need to embed positions
        # NOTE(Alex): hacky... assumes pad idx is 0
        # TODO(Alex): is using src_masks as input to positional embeddings correct?
        x = src_embs + self.embed_positions(src_masks.long())
        #x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # used to mask padding in input
        #encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B
        #encoder_padding_mask = (1. - src_masks).byte().t() # -> B x T
        encoder_padding_mask = (1. - src_masks).byte() # B x T
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)
        # B x T x C -> T x B x C
        #x = x.transpose(0, 1)

        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(1), 0)
                #x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=1)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # B x C x T -> B x T x C
        x = x.transpose(2, 1)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            #encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        # scale gradients (this only affects backward, not forward)
        # self.num_attention_layers in their code is set by the overall FConvModel
        #x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        #y = (x + input_embedding) * math.sqrt(0.5)

        #return {
        #    'encoder_out': (x, y),
        #    'encoder_padding_mask': encoder_padding_mask,  # B x T
        #}
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = (
                encoder_out['encoder_out'][0].index_select(0, new_order),
                encoder_out['encoder_out'][1].index_select(0, new_order),
            )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.input_dim # gets projected back to inp dim


#class FConvDecoder(FairseqIncrementalDecoder):
class FConvDecoder(nn.Module):
    """Convolutional decoder"""

    def __init__(
            self, vocab, input_dim=512, output_dim=512,
            max_positions=1024, convolutions=((512, 3),) * 20,
            dropout=0.1,
            left_pad=False,
    ):
        super().__init__()
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout
        self.left_pad = left_pad
        self.input_dim = input_dim

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]

        padding_idx = vocab.get_token_index(vocab._padding_token)
        self.embed_positions = PositionalEmbedding(max_positions, input_dim, padding_idx,
                                                   left_pad=self.left_pad)

        self.fc1 = Linear(input_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            self.convolutions.append(
                Conv1d(in_channels, out_channels * 2, kernel_size,
                       padding=(kernel_size - 1), dropout=dropout))
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.fc2 = Linear(in_channels, output_dim)

    #def forward(self, prev_output_tokens, encoder_out_dict=None, incremental_state=None):
    def forward(self, tok_embs, src_masks):
        """

        Args:
            - tok_embs: token embeddings, size (batch size x seq len x d_emb)
            - src_masks: source mask

        Returns:

        """

        # embed tokens and combine with positional embeddings
        pos_embs = self.embed_positions(src_masks.long())
        src_embs = tok_embs + pos_embs
        src_embs = F.dropout(src_embs, p=self.dropout, training=self.training)

        # project to size of convolution
        x = self.fc1(src_embs)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # temporal convolutions
        residuals = [x]
        for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)[:,:, :-conv.padding[0]] # need to drop future steps
            x = F.glu(x, dim=1)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # B x C x T -> B x T x C
        x = x.transpose(2, 1)
        x = self.fc2(x)
        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions() if self.embed_positions is not None else float('inf')

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.input_dim # gets projected back to inp dim

def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:
            positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1

def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalizedf Conv1d layer"""
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""

    class ConvTBC(torch.nn.Module):
        """1D convolution over an input of shape (time x batch x channel)
        The implementation uses gemm to perform the convolution. This implementation
        is faster than cuDNN for small kernel sizes.
        """
        def __init__(self, in_channels, out_channels, kernel_size, padding=0):
            super(ConvTBC, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = _single(kernel_size)
            self.padding = _single(padding)

            self.weight = torch.nn.Parameter(torch.Tensor(
                self.kernel_size[0], in_channels, out_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        def forward(self, input):
            return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding[0])

        def __repr__(self):
            s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
                 ', padding={padding}')
            if self.bias is None:
                s += ', bias=False'
            s += ')'
            return s.format(name=self.__class__.__name__, **self.__dict__)

    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
