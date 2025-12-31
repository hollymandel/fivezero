from typing import List

import torch
import torch.nn as nn

from fivezero.gameEngine import Actor, N, State


class ConvNet(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        # two input channels - current player and opponent
        self.conv1 = torch.nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu4 = torch.nn.ReLU()
        self.fc_policy = torch.nn.Linear(
            32 * N**2, N**2
        )  # P(s,a) logit for prob of each move a given s
        self.fc_value = torch.nn.Linear(32 * N**2, 1)  # V(s) value given s
        self.device = device

        # move to device
        self.to(device)

    def encode(self, x: State):
        """Board encoding such that first channel is next-to-play (State.player) stones, second channel
        is opponent's stones. returns board as 1x2x5x5, stack batch along dimension 0"""
        if x.player == Actor.POSITIVE:
            x_board = torch.tensor(
                x.board == 1, dtype=torch.float32, device=self.device
            )
            x_opponent = torch.tensor(
                x.board == -1, dtype=torch.float32, device=self.device
            )
        else:
            x_board = torch.tensor(
                x.board == -1, dtype=torch.float32, device=self.device
            )
            x_opponent = torch.tensor(
                x.board == 1, dtype=torch.float32, device=self.device
            )
        x = torch.stack([x_board, x_opponent], dim=0)
        # add batch dimension
        x = x.unsqueeze(0)
        return x

    def internal_forward(self, x):
        # skip connections at each layer after first
        res = self.relu1(self.conv1(x))
        res = self.relu1(self.conv2(res)) + res
        res = self.relu1(self.conv3(res)) + res
        res = self.relu1(self.conv4(res)) + res
        return res.view(res.size(0), -1)  # (batch_size, 32 * N**2)

    def forward_policy(self, x):
        last_hidden = self.internal_forward(x)
        # return nn.Softmax(dim=1)(self.fc_policy(last_hidden))
        return self.fc_policy(last_hidden)

    def forward_value(self, x):
        last_hidden = self.internal_forward(x)
        return nn.Tanh()(self.fc_value(last_hidden))

    def forward(self, x):
        return self.forward_policy(x), self.forward_value(x)
