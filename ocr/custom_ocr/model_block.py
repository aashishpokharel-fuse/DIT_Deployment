from torch.nn import Module, ModuleList
from torch.nn import Conv2d, InstanceNorm2d, Dropout, Dropout2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d, Linear, Sigmoid, ReLU, MaxPool2d
from torch.nn import ReLU
from torch.nn.functional import pad
import random
import torch 
import torch.nn as nn
import math
from torch.nn import functional as f
import numpy as np

class CharModel:
    def __init__(self):

        self.char2index = {}
        self.index2char = {}
        self.n_chars = 3
        
        self.char2index['PAD'] = 0
        self.char2index['TSOS'] = 1
        self.char2index['TEOS'] = 2

        self.index2char[0] = 'PAD'
        self.index2char[1] = 'TSOS'
        self.index2char[2] = 'TEOS'

        self.char2lm = []
        # self.special_tokens_in_vocab = ['TSOS', 'TEOS']
        # self.special_tokens_out_vocab = ['BLANK', 'PAD']

    def add_char_collection(self, char_collection):
        for char in char_collection:
            self.add_char(char)

        self.vocab_size = self.n_chars

        # for t in self.special_tokens_in_vocab:
        #     self.add_char(t)

        # for ot in self.special_tokens_out_vocab:
        #     self.add_char(ot)

        # self.vocab_size = self.n_chars - len(self.special_tokens_out_vocab)

    def add_char(self, char):

        if char not in self.char2index:
            # char_index = str.encode(char)[0] + 2
            char_index = self.n_chars
            self.char2index[char] = char_index
            self.index2char[char_index] = char
            self.n_chars += 1

    def char2lm_mapping(self, index, char):
        # return index
        if index > 3:
            new_index = str.encode(char)
            if len(new_index) == 1:
                return new_index[0] + 2
            else:
                print(index, char)
                raise ValueError("Cannot convert character to token")
            
        else:
            return index

    def __call__(self, char_collection):
        self.add_char_collection(char_collection)

        for i, c in self.index2char.items():
            self.char2lm.append(self.char2lm_mapping(i, c))
            if i>=93:
                break


    def indexes2characters(self, indexes, ctc_mode=False):
        characters = []
        if not ctc_mode:
            for i in indexes:
                characters.append(self.index2char[i])
            return characters
        else:
            for j, i in enumerate(indexes):
                if i == self.n_chars:
                    continue
                if characters and indexes[j-1] == i:
                    continue
                characters.append(self.index2char[i])
            return characters


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(128.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :]


class SEBlock(Module):
    def __init__(self, in_channels, reduction) -> None:
        super(SEBlock, self).__init__()
        self.gap = AdaptiveAvgPool2d(1)
        self.linear1 = Linear(in_channels, in_channels//reduction, bias=False)
        self.relu = ReLU()
        self.linear2 = Linear(in_channels//reduction, in_channels, bias=False)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        x1 = self.gap(x).squeeze(2).squeeze(2)        
        x1 = self.relu(self.linear1((x1)))
        x1 = self.sigmoid(self.linear2((x1)))[:, :, None, None]
        
        output = x1 * x

        return output


class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


class FCN_Encoder_SE(Module):
    def __init__(self, params):
        super(FCN_Encoder_SE, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            SEBlock(16, 16),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            SEBlock(32, 16),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            SEBlock(64, 16),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            SEBlock(128, 16),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            SEBlock(128, 16),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            SEBlock(128, 16),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, pool=(1, 1), dropout=self.dropout),
        ])
        self.se_blocks = ModuleList([
            SEBlock(128, 16),
            SEBlock(128, 16),
            SEBlock(128, 16),
            SEBlock(256, 16),
        ])

        # self.ada_pool = AdaptiveMaxPool2d((1, None))
        self.ada_pool = MaxPool2d(kernel_size=(2,1))


    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b, se in zip(self.blocks, self.se_blocks):
            xt = se(b(x))
            x = x + xt if x.size() == xt.size() else xt

        x = self.ada_pool(x)

        return x


class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1, 1), k=3, activation=ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.norm_layer.eval()
        
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class DSCBlock(Module):

    def __init__(self, in_, out_, pool=(2, 1), activation=ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=pool)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)
        self.norm_layer.eval()
    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class ImgCharModel:
    def __init__(self):
        self.char2index = {}
        self.index2char = {}
        self.n_chars = 2

        self.char2index['ISOS'] = 0
        self.char2index['IEOS'] = 1
        
        self.index2char[0] = 'ISOS'
        self.index2char[1] = 'IEOS'


class ImageEncoder(nn.Module):

    def __init__(self, pos_encoding, config, device, vocab_size):
        super(ImageEncoder, self).__init__()
        # """

        self.pos_encoding = pos_encoding

        params = {"dropout": 0.5, "input_channels": 3}
        self.resnet = FCN_Encoder_SE(params)

        self.device = device
        img_char_model = ImgCharModel()

        sos_token = img_char_model.char2index['ISOS']
        self.sos_token = torch.LongTensor([[sos_token]])
        eos_token = img_char_model.char2index['IEOS']
        self.eos_token = torch.LongTensor([[eos_token]])

        self.img_embedding = nn.Embedding(img_char_model.n_chars, config.char_embedding_dim)

        # self.linear_projector = nn.Linear(256, 512)
        self.aux_linear = nn.Linear(config.char_embedding_dim, vocab_size+1)

        self.cls_classifier = nn.Linear(config.char_embedding_dim, 8)


    def forward(self, src, **kwargs):
        aux_ctc = kwargs.pop("aux_ctc", False)

        # """
        # print("SRC:", src.shape)

        char_embedding = self.resnet(src)

        cls_embedding = f.adaptive_avg_pool2d(char_embedding, (1, 1)).squeeze(-1).squeeze(-1)
        cls_classification = self.cls_classifier(cls_embedding)
        # print("CLS Embedding:", cls_embedding.shape)

        # print("Char embedding:", char_embedding.shape)
        char_embedding = char_embedding.squeeze(dim=-2).permute(0, 2, 1)


        bs = src.shape[0]
        sos_token = self.img_embedding(self.sos_token.to(self.device))
        sos_token = sos_token.repeat(bs, 1, 1)
        eos_token = self.img_embedding(self.eos_token.to(self.device))
        eos_token = eos_token.repeat(bs, 1, 1)
        char_embedding = torch.cat([sos_token, char_embedding, eos_token], axis=1)
        char_embedding =(char_embedding + self.pos_encoding(char_embedding))
        # char_embedding_pe =(char_embedding + self.pos_encoding(char_embedding))
        
        if aux_ctc:
            aux_features = self.aux_linear(char_embedding)
            aux_features = aux_features.permute(1,0,2).contiguous().log_softmax(2)

            return cls_classification, cls_embedding, char_embedding, aux_features

        return cls_classification, cls_embedding, char_embedding

class TextDecoder(nn.Module):

    def __init__(self, vocab_size, char_embedding, pos_encoding, config, device):
        super(TextDecoder, self).__init__()

        self.device = device
        self.char_embedding = char_embedding
        self.pos_encoding = pos_encoding

        self.dropout = nn.Dropout(p=config.transformer_decoder['dropout'])

        decoder_layer = nn.TransformerEncoderLayer(d_model=config.char_embedding_dim,
                                                   nhead=config.transformer_decoder['num_heads'],
                                                   dropout=config.transformer_decoder['dropout'],
                                                   dim_feedforward=config.transformer_decoder['ffn'],
                                                   activation='relu')
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer,
                                                         num_layers=config.transformer_decoder['num_layers'])

        # self.linear = nn.Linear(config.char_embedding_dim, vocab_size)
        embed_shape = self.char_embedding.weight.shape
        self.linear = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.linear.weight = self.char_embedding.weight # Tied weights

        self.cls_classifier = nn.Linear(config.char_embedding_dim, 2)

        self.aux_linear = nn.Linear(config.char_embedding_dim, vocab_size+1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, memory, tgt, tgt_mask, cls_embedding, first_seq_len, src_key_padding_mask=None ):
        tgt = self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))

        tgt = torch.cat([cls_embedding, memory, tgt], axis=1)
        tgt = tgt.permute(1, 0, 2)
        
        output = self.transformer_decoder(
            src=tgt,
            mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, 
        )

        output = output.permute(1, 0, 2)

        output = output[:,first_seq_len,:]

        output = self.linear(output)
        output = self.softmax(output)
        
        return output
