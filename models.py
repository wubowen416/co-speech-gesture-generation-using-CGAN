import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, i_size, n_size, o_size, h_size):

        super(Generator, self).__init__()

        self.i_size = i_size
        self.n_size = n_size
        self.o_size = o_size

        self.i_fc = nn.Linear(i_size, int(h_size/2))
        self.n_fc = nn.Linear(n_size, int(h_size/2))
        self.o_fc = nn.Linear(2 * h_size, o_size)
        self.rnn = nn.LSTM(h_size, h_size, num_layers=2, bidirectional=True)

    def forward(self, i, n):
        """3D tensor"""

        assert len(
            i.shape) == 3, f"expect 3D tensor with shape (t, n, dim), got {i.shape}"
        assert n.size(
            0) == 1, f"shape of noise must be (1, N, dim), got noise with {n.size(0)}"
        assert i.size(1) == n.size(
            1), f"batch size of input and noise must be the same, got input with {i.size(1)}, noise with {n.size(1)}"

        n = n.repeat(i.size(0), 1, 1)
        t, bs = i.size(0), i.size(1)
        i = i.view(t*bs, -1)
        n = n.view(t*bs, -1)
        i = F.leaky_relu(self.i_fc(i), 1e-2)
        n = F.leaky_relu(self.n_fc(n), 1e-2)
        i = i.view(t, bs, -1)
        n = n.view(t, bs, -1)
        x = torch.cat([i, n], dim=-1)
        x, _ = self.rnn(x)
        x = F.leaky_relu(x, 1e-2)
        o = self.o_fc(x)
        return o

    def forward_given_noise_seq(self, i, n):

        t, bs = i.size(0), i.size(1)
        i = i.view(t*bs, -1)
        n = n.view(t*bs, -1)
        i = F.leaky_relu(self.i_fc(i), 1e-2)
        n = F.leaky_relu(self.n_fc(n), 1e-2)
        i = i.view(t, bs, -1)
        n = n.view(t, bs, -1)
        x = torch.cat([i, n], dim=-1)
        x, _ = self.rnn(x)
        x = F.leaky_relu(x, 1e-2)
        o = self.o_fc(x)
        return o

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class Discriminator(nn.Module):

    def __init__(self, i_size, c_size, h_size):

        super(Discriminator, self).__init__()

        self.i_size = i_size
        self.c_szie = c_size

        self.i_fc = nn.Linear(i_size, int(h_size/2))
        self.c_fc = nn.Linear(c_size, int(h_size/2))
        self.rnn = nn.LSTM(h_size, h_size, num_layers=2, bidirectional=True)
        self.o_fc = nn.Linear(2*h_size, 1)

    def forward(self, x, c):
        """3D tensor + 3D tensor"""

        assert len(x.shape) == 3, f"expect 3D tensor, got {x.shape}"

        t, bs = x.size(0), x.size(1)
        x = x.view(t*bs, -1)
        c = c.view(t*bs, -1)
        x = F.leaky_relu(self.i_fc(x))
        c = F.leaky_relu(self.c_fc(c))
        x = x.view(t, bs, -1)
        c = c.view(t, bs, -1)
        x = torch.cat([x, c], dim=-1)
        x, _ = self.rnn(x)
        x = x.view(t*bs, -1)
        x = torch.sigmoid(self.o_fc(x))
        x = x.view(t, bs, -1)
        return x

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
