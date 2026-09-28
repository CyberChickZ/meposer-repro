from torch import nn

from .layers import mlp


class IMUBranch(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        self.net = nn.Sequential(mlp([in_dim, cfg.hidden, cfg.out_dim], final_act=True), nn.LayerNorm(cfg.out_dim))
        self.out_dim = cfg.out_dim

    def forward(self, x):
        return self.net(x)
