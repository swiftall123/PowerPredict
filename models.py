import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,n_lags,input_size,hidden_layer_size,num_layers,output_size,dropout):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear_1 = nn.Linear(input_size,hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size,hidden_size=self.hidden_layer_size
                            ,num_layers=num_layers,batch_first=True
                            )
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(n_lags*hidden_layer_size,output_size)
    def init_weights(self):
        for name,param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param,0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self,x):
        batchsize = x.shape[0]
        x = self.linear_1(x)
        x = self.relu(x)

        lstm_out,(h_n,c_n)=self.lstm(x)
        x = lstm_out.reshape(batchsize,-1)
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embed,n_head,dropout_p):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads
        self.n_embed = n_embed
        self.dropout_p = dropout_p
        self.n_head = n_head

        self.key = nn.Linear(self.n_embed, self.n_embed)
        self.query = nn.Linear(self.n_embed, self.n_embed)
        self.value = nn.Linear(self.n_embed, self.n_embed)
        # regularization
        self.dropout = nn.Dropout(self.dropout_p)
        # output projection
        self.proj = nn.Linear(self.n_embed, self.n_embed)

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.dropout(self.proj(y))
        return y, att


class FeedForward(nn.Module):
    def __init__(self, n_embed,mlp_dim,dropout_p):
        super().__init__()
        dim = n_embed
        hidden_dim = mlp_dim
        dropout_p = dropout_p
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self,n_embed,n_head,dropout_p,mlp_dim):
        super(Block, self).__init__()

        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = SelfAttention(n_embed,n_head,dropout_p)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward( n_embed,mlp_dim,dropout_p)

    def forward(self, x):
        h, att_map = self.attn(self.ln1(x))
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, att_map


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        n_embed = config.n_embed
        n_layers = config.n_layers
        fourier_dim = sum(config.ks) * 2
        num_tokens = (config.n_lags - config.patch_size) // config.stride + 1
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.to_embedding = nn.Linear(config.patch_size, n_embed)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, n_embed))
        self.transformer = nn.ModuleList([Block(config) for _ in range(n_layers)])
        self.head = nn.Linear(n_embed * num_tokens, config.period)
        self.revin = RevIN()
        self.fourier = FourierFeatures(config.ks, config.ps)
        self.inr = nn.Sequential(nn.Linear(fourier_dim, 128), nn.ReLU(True),
                                 nn.Linear(128, 128), nn.ReLU(True),
                                 nn.Linear(128, 128), nn.ReLU(True),
                                 nn.Linear(128, 128), nn.ReLU(True),
                                 nn.Linear(128, 128), nn.ReLU(True),
                                 nn.Linear(128, 1))

    def forward(self, x, time):  # x: [b, n_lags]
        att_maps = []
        # x = self.revin(x, 'norm')
        b, c = x.shape
        seasonal = self.inr(self.fourier(time)).squeeze(2)
        x = self.to_embedding(x.unfold(dimension=1, size=self.patch_size, step=self.stride))
        x = x + self.pos_embedding
        for block in self.transformer:
            x, att_map = block(x)
        x = self.head(x.flatten(1)) + seasonal
        # x = self.revin(x, 'denorm')
        return x


class RevIN(nn.Module):
    def __init__(self, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

    def _get_statistics(self, x):
        if self.subtract_last:
            self.last = x[:, -1].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class FourierFeatures(nn.Module):
    def __init__(self, ks, ps):  # [daily, weakly, yearly]
        super().__init__()
        terms = []
        for k, p in zip(ks, ps):
            term = 2 * torch.pi * torch.arange(1, k + 1) / p
            terms.append(term)
        self.term = torch.cat(terms, dim=0).unsqueeze(0).unsqueeze(0)

    def forward(self, ts):  # [b, period, 1]
        self.term = self.term.to(ts.device)
        basis = self.term * ts
        fourier = torch.cat([torch.sin(basis), torch.cos(basis)], dim=-1)  # [b, period, terms]
        return fourier


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, dropout_p=0.2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, num_tokens, mlp_token_dim, dropout_p=0.2):
        super(MixerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(num_tokens)
        self.channel_mix = MLP(dim, mlp_dim, dropout_p)
        self.token_mix = MLP(num_tokens, mlp_token_dim, dropout_p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.token_mix(self.ln2(x)) + x
        x = x.permute(0, 2, 1)
        x = self.channel_mix(self.ln1(x)) + x
        return x


class MLPMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed = config.n_embed
        n_layers = config.n_layers
        mlp_dim = config.mlp_dim
        mlp_token_dim = 128
        num_tokens = (config.n_lags - config.patch_size) // config.stride + 1
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.to_embedding = nn.Linear(config.patch_size, n_embed)
        self.head = nn.Linear(n_embed * num_tokens, config.period)
        self.revin = RevIN()
        self.mixer = nn.ModuleList(
            [MixerBlock(n_embed, mlp_dim, num_tokens, mlp_token_dim, config.dropout_p) for _ in range(n_layers)])
        self.residual = nn.Linear(config.n_lags, config.period)

    def forward(self, x):  # x: [b, n_lags]
        x = self.revin(x, 'norm')
        shortcut = x
        b, c = x.shape
        x = self.to_embedding(x.unfold(dimension=1, size=self.patch_size, step=self.stride))
        for block in self.mixer:
            x = block(x)
        x = self.head(x.flatten(1))
        # x = x + self.residual(shortcut)
        x = self.revin(x, 'denorm')
        return x


class VMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = 512
        fourier_dim = sum(config.ks) * 2
        n_layres = config.n_layers
        self.nets = nn.ModuleList([MLP(dim, dim, config.dropout_p) for _ in range(n_layres)])
        self.lns = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layres)])
        self.to_embedding = nn.Linear(config.n_lags, dim)
        self.head = nn.Linear(dim, config.period)
        self.revin = RevIN()
        self.fourier = FourierFeatures(config.ks, config.ps)
        self.inr = nn.Sequential(nn.Linear(fourier_dim, 128), nn.ReLU(True),
                                 nn.Linear(128, 128), nn.ReLU(True),
                                 nn.Linear(128, 128), nn.ReLU(True),
                                 nn.Linear(128, 128), nn.ReLU(True),
                                 nn.Linear(128, 128), nn.ReLU(True),
                                 nn.Linear(128, 1))

    def forward(self, x, time):  # x:[b, n_lags]  time:[b, period]
        # x = self.revin(x, 'norm')
        seasonal = self.inr(self.fourier(time)).squeeze(2)
        x = self.to_embedding(x)
        for block, ln in zip(self.nets, self.lns):
            x = block(ln(x)) + x
        x = self.head(x) + seasonal
        # x = self.revin(x, 'denorm')
        return x


class VMLP2(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = 512
        fourier_dim = sum(config.ks) * 2
        n_layres = config.n_layers
        self.nets = nn.ModuleList([MLP(dim, dim, config.dropout_p) for _ in range(n_layres)])
        self.lns = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layres)])

        self.to_embedding = nn.Linear(config.n_lags, dim)
        self.head = nn.Linear(dim, 1)
        self.revin = RevIN()
        self.fourier = FourierFeatures(config.ks, config.ps)
        self.proj_date = nn.Linear(fourier_dim, dim)
        self.resiual = nn.Linear(config.n_lags, config.period)
        self.positional_encoding = nn.Parameter(torch.randn(1, config.period, dim))

    def forward(self, x, time):  # x:[b, n_lags]  time:[b, period]
        # x = self.revin(x, 'norm')
        date_embedding = self.proj_date(self.fourier(time))
        x = self.to_embedding(x).unsqueeze(1) + self.positional_encoding + date_embedding
        for block, ln in zip(self.nets, self.lns):
            x = block(ln(x)) + x
        x = self.head(x).squeeze(2)
        # x = self.revin(x, 'denorm')
        return x


class PatchRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = 4
        self.stride = 4
        self.rnn = nn.LSTM(input_size=self.patch_size, hidden_size=128, num_layers=3, batch_first=True)
        self.out = nn.Linear(128, config.period)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, time):  # x:[b, n_lags]
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        o, (h, c) = self.rnn(x)
        out = self.out(h[-1])
        return out


class PointTransformer(nn.Module):
    def __init__(self, n_embed,n_layers,num_tokens,period,n_head,dropout_p,mlp_dim):
        super(PointTransformer, self).__init__()
        # n_embed = config.n_embed
        # n_layers = config.n_layers
        # num_tokens = config.n_lags
        self.to_embedding = nn.Linear(15, n_embed)   # 15个通道
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, n_embed))
        self.transformer = nn.ModuleList([Block(n_embed,n_head,dropout_p,mlp_dim) for _ in range(n_layers)])
        self.head = nn.Linear(n_embed, period)
        self.revin = RevIN()

    def forward(self, x):  # x: [b, n_lags]
        # x = self.revin(x, 'norm')
        x = self.to_embedding(x)
        x = x + self.pos_embedding
        for block in self.transformer:
            x, _ = block(x)
        x = self.head(x[:, -1])
        return x


class PointTransformer3(nn.Module):
    def __init__(self, config):
        super(PointTransformer3, self).__init__()
        n_embed = config.n_embed
        n_layers = config.n_layers
        num_tokens = config.n_lags
        self.to_embedding = nn.Linear(5, n_embed)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, n_embed))
        self.transformer = nn.ModuleList([Block(config) for _ in range(n_layers)])
        self.head = nn.Linear(n_embed * num_tokens, config.period)
        self.revin = RevIN()

    def forward(self, x):  # x: [b, n_lags]
        # x = self.revin(x, 'norm')
        x = self.to_embedding(x)
        x = x + self.pos_embedding
        for block in self.transformer:
            x, _ = block(x)
        x = self.head(x.flatten(1))
        return x


class PointTransformer2(nn.Module):
    def __init__(self, config):
        super(PointTransformer2, self).__init__()
        n_embed = config.n_embed
        n_layers = config.n_layers
        num_tokens = config.n_lags
        self.period = config.period
        self.to_embedding = nn.Linear(1, n_embed)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, n_embed))
        self.transformer = nn.ModuleList([Block(config) for _ in range(n_layers)])
        self.head = nn.Conv1d(self.period, self.period, kernel_size=n_embed, stride=1, groups=self.period)
        self.revin = RevIN()

    def forward(self, x, time):  # x: [b, n_lags]
        att_maps = []
        b, c = x.shape
        x = self.to_embedding(x.unfold(dimension=1, size=1, step=1))
        x = x + self.pos_embedding
        for block in self.transformer:
            x, att_map = block(x)
            att_maps.append(att_map)
        x = torch.cat([x[:, -i:].mean(dim=1, keepdim=True) for i in range(1, self.period + 1)], dim=1)
        x = self.head(x).squeeze(2)  # x: [b, n_lags, n_embed]
        return x