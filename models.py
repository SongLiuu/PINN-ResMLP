import torch
import torch.nn as nn



def act_func(name = 'relu'):
    # 默认使用 ReLU
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'silu': nn.SiLU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'mish': nn.Mish(),
        'sigmoid': nn.Sigmoid(),
        'tanhshrink': nn.Tanhshrink(),
        'elu': nn.ELU(),
    }
    return activations.get(name, nn.ReLU())


class MLP(nn.Module):
    def __init__(self, varb_num, dropout = 0.3, act_name = 'relu'):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Linear(varb_num, 96),
            act_func(act_name), nn.Dropout(dropout),
            nn.Linear(96, 192),
            act_func(act_name), nn.Dropout(dropout),
            nn.Linear(192, 384),
            act_func(act_name), nn.Dropout(dropout),
            nn.Linear(384, 768),
            act_func(act_name), nn.Dropout(dropout),
            )
        self.linear_out = nn.Sequential(
            nn.Linear(768, 1),
            # nn.ReLU()
            )

    def forward(self, x):
        x = self.feature_extract(x)
        return self.linear_out(x)


class LinearResnet_small(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, act_name = 'relu'):
        super().__init__()
        self.res = nn.Sequential(
            act_func(act_name), nn.Dropout(dropout),
            nn.Linear(in_channel, out_channel), 
            act_func(act_name), nn.Dropout(dropout),
            nn.Linear(out_channel, out_channel),
            )
        self.linear = nn.Linear(in_channel, out_channel)
    
    def forward(self, x):
        return self.linear(x) + self.res(x)


class ResnetMLP_small(nn.Module):
    def __init__(self, varb_num, dropout, act_name = 'relu'):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Linear(varb_num, 96),
            LinearResnet_small(96, 192, dropout, act_name = act_name),
            LinearResnet_small(192, 384, dropout, act_name = act_name),
            # LinearResnet(384, 384, dropout),
            LinearResnet_small(384, 768, dropout, act_name = act_name),
            # LinearResnet(768, 768, dropout),
            act_func(act_name), nn.Dropout(dropout),
            )
        self.linear_out = nn.Sequential(
            nn.Linear(768, 1),
            # nn.ReLU()
            )

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.linear_out(x)
        return x


class LinearResnet_large(nn.Module):
    def __init__(self, in_channel, dropout, act_name = 'relu'):
        super().__init__()
        self.res = nn.Sequential(
            act_func(act_name), nn.Dropout(dropout),
            nn.Linear(in_channel, in_channel), 
            act_func(act_name), nn.Dropout(dropout),
            nn.Linear(in_channel, in_channel),
            )
    
    def forward(self, x):
        return x + self.res(x)


class ResnetMLP_large(nn.Module):
    def __init__(self, varb_num, dropout, act_name = 'relu'):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Linear(varb_num, 64),
            LinearResnet_large(64, dropout, act_name = act_name),
            nn.Linear(64, 128),
            LinearResnet_large(128, dropout, act_name = act_name),
            nn.Linear(128, 256),
            LinearResnet_large(256, dropout, act_name = act_name),
            nn.Linear(256, 512),
            LinearResnet_large(512, dropout, act_name = act_name),
            nn.Linear(512, 1024),
            LinearResnet_large(1024, dropout, act_name = act_name),
            act_func(act_name), nn.Dropout(dropout),
            )
        self.linear_out = nn.Sequential(
            nn.Linear(1024, 1),
            # nn.ReLU()
            )

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.linear_out(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_channel, hidden_channel, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_channel, in_channel),
            nn.Dropout(dropout)
            )
    
    def forward(self, x):
        x = self.net(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel, dropout = 0.):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_channel),
            FeedForward(in_channel, hidden_channel, dropout)
            )
    
    def forward(self, x):
        return x + self.block(x)


class StackFeedForward(nn.Module):
    def __init__(self, varb_num, dropout = 0., act_name = 'relu'):
        super().__init__()
        self.linear_in = nn.Sequential(
            nn.Linear(varb_num, 768),
            nn.GELU()
            )
        
        self.feature_extract = nn.ModuleList([])
        for i in range(4):
            self.feature_extract.append(MixerBlock(768, 196))
        
        self.layernorm = nn.LayerNorm(768)
        self.linear_out = nn.Sequential(
            nn.Linear(768, 1),
            # nn.ReLU()
            )

    def forward(self, x):
        x = self.linear_in(x)
        for block in self.feature_extract:
            x = block(x)
        x = self.layernorm(x)
        x = self.linear_out(x)
        return x


def set_parameter_not_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def get_model(model, x_features, dropout, act_name = 'relu'):
    if model == 'MLP':
        return MLP(x_features, dropout, act_name = act_name)
    elif model == 'ResnetMLP_small':
        return ResnetMLP_small(x_features, dropout, act_name = act_name)
    elif model == 'ResnetMLP_large':
        return ResnetMLP_large(x_features, dropout, act_name = act_name)
    elif model == 'StackFeedForward':
        return StackFeedForward(x_features, dropout, act_name = act_name)




