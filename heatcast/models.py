import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rnn = nn.LSTM(channels, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1]).squeeze(-1)


class ConvLSTM(nn.Module):
    def __init__(self, channels, hidden=12):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(channels + hidden, 4 * hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        h = x.new_zeros((len(x), self.hidden, *x.shape[-2:]))
        c = torch.zeros_like(h)
        for step in x.unbind(1):
            i, f, o, g = self.gates(torch.cat([step, h], 1)).chunk(4, 1)
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            h = o.sigmoid() * c.tanh()
        return self.head(h).squeeze(1)
