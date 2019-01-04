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
            self, vocab, embed_dim=512, embed_dict=None, max_positions=1024,
            convolutions=((512, 3),) * 20, dropout=0.1, left_pad=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.left_pad = left_pad
        self.num_attention_layers = None
        self.input_dim = embed_dim

        #num_embeddings = vocab.get_vocab_size()
        self.padding_idx = vocab.get_token_index(vocab._padding_token)
        #self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        #if embed_dict:
        #    self.embed_tokens = utils.load_embedding(embed_dict, self.vocab, self.embed_tokens)
        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
            left_pad=self.left_pad,
        )

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
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
        self.fc2 = Linear(in_channels, embed_dim)

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
        #import ipdb; ipdb.set_trace()
        # embed tokens and positions
        #x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #input_embedding = x
        # NOTE(Alex): hacky... assumes pad idx is 0
        # TODO(Alex): is using src_masks as input to positional embeddings correct?
        x = src_embs + self.embed_positions(src_masks.long())
        input_embedding = x
        if x is None:
            import ipdb
            ipdb.set_trace()

        #import ipdb; ipdb.set_trace()

        # project to size of convolution
        x = self.fc1(x)

        # used to mask padding in input
        #encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B
        encoder_padding_mask = (1. - src_masks).byte() # B x T
        #encoder_padding_mask = (1. - src_masks).byte().t() # -> B x T
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
            #x = F.glu(x, dim=2)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # B x C x T -> B x T x C
        x = x.transpose(2, 1)
        # T x B x C -> B x T x C
        #x = x.transpose(1, 0)

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


'''
class FConvDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""

    def __init__(
            self, dictionary, embed_dim=512, embed_dict=None, out_embed_dim=256,
            max_positions=1024, convolutions=((512, 3),) * 20, attention=True,
            dropout=0.1, share_embed=False, positional_embeddings=True,
            adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0,
            left_pad=False,
    ):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout
        self.left_pad = left_pad
        self.need_attn = True

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(embed_dict, self.dictionary, self.embed_tokens)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
            left_pad=self.left_pad,
        ) if positional_embeddings else None

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
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
                LinearizedConv1d(in_channels, out_channels * 2, kernel_size,
                                 padding=(kernel_size - 1), dropout=dropout)
            )
            self.attention.append(AttentionLayer(out_channels, embed_dim)
                                  if attention[i] else None)
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.adaptive_softmax = None
        self.fc2 = self.fc3 = None

        if adaptive_softmax_cutoff is not None:
            assert not share_embed
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, in_channels, adaptive_softmax_cutoff,
                                                    dropout=adaptive_softmax_dropout)
        else:
            self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, \
                    "Shared embed weights implies same dimensions " \
                    " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self, prev_output_tokens, encoder_out_dict=None, incremental_state=None):
        if encoder_out_dict is not None:
            encoder_out = encoder_out_dict['encoder_out']
            encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

            # split and transpose encoder outputs
            encoder_a, encoder_b = self._split_encoder_out(encoder_out, incremental_state)

        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
        else:
            pos_embed = 0

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self._embed_tokens(prev_output_tokens, incremental_state)

        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for proj, conv, attention, res_layer in zip(self.projections, self.convolutions, self.attention,
                                                    self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)

                x, attn_scores = attention(x, target_embedding, (encoder_a, encoder_b), encoder_padding_mask)

                if not self.training and self.need_attn:
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

                x = self._transpose_if_training(x, incremental_state)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x, incremental_state)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc3(x)

        return x, avg_attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        encoder_out = utils.get_incremental_state(self, incremental_state, 'encoder_out')
        if encoder_out is not None:
            encoder_out = tuple(eo.index_select(0, new_order) for eo in encoder_out)
            utils.set_incremental_state(self, incremental_state, 'encoder_out', encoder_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions() if self.embed_positions is not None else float('inf')

    def upgrade_state_dict(self, state_dict):
        if utils.item(state_dict.get('decoder.version', torch.Tensor([1]))[0]) < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['decoder.version'] = torch.Tensor([1])
        return state_dict

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _split_encoder_out(self, encoder_out, incremental_state):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = utils.get_incremental_state(self, incremental_state, 'encoder_out')
        if cached_result is not None:
            return cached_result

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)

        if incremental_state is not None:
            utils.set_incremental_state(self, incremental_state, 'encoder_out', result)
        return result

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

class FConvLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if hasattr(args, 'max_target_positions'):
            args.tokens_per_sample = args.max_target_positions

        decoder = FConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.tokens_per_sample,
            share_embed=False,
            positional_embeddings=False,
            adaptive_softmax_cutoff=(
                eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            adaptive_softmax_dropout=args.adaptive_softmax_dropout,
        )
        return FConvLanguageModel(decoder)
'''

def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]

def base_lm_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(1268, 4)] * 13')
    args.decoder_attention = getattr(args, 'decoder_attention', 'False')
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)

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
