from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, dim, ffn_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        y, state = self.lstm(x, state)
        x = self.norm1(x + self.drop(y))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, state


class TemporalEncoder(nn.Module):
    def __init__(self, dim, blocks, ffn_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([TemporalBlock(dim, ffn_dim, dropout) for _ in range(blocks)])

    def forward(self, x, states=None):
        states = states or [None] * len(self.blocks)
        new_states = []
        for block, s in zip(self.blocks, states):
            x, s = block(x, s)
            new_states.append(s)
        return x, new_states
