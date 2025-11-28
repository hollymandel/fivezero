import torch
import torch.nn as nn
from typing import List
from FiveZero.GameEngine import N, State

class ConvNet(torch.nn.Module):
    def __init__(self, device: torch.device):
        super(ConvNet, self).__init__()
        # two input channels - current player and opponent
        self.conv1 = torch.nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu4 = torch.nn.ReLU()
        self.fc_policy = torch.nn.Linear(32 * N**2, N**2) # prob of each move 
        self.fc_value = torch.nn.Linear(32 * N**2, N**2) # value, if each move taken 
        self.device = device

        # move to device
        self.to(device)

    def encode(self, x: State):
        x_board = torch.tensor(x.board == 1, dtype=torch.float32, device=self.device)
        x_opponent = torch.tensor(x.board == -1, dtype=torch.float32, device=self.device)
        x = torch.stack([x_board, x_opponent], dim=0)
        return x

    def forward(self, x: List[State]):
        # process to a 2x5x5 tensor, one channel for each player
        x = [self.encode(x_i) for x_i in x]
        x = torch.stack(x, dim=0)

        # skip connections at each layer after first
        res = self.relu1(self.conv1(x))
        res = self.relu1(self.conv2(res)) + res
        res = self.relu1(self.conv3(res)) + res
        res = self.relu1(self.conv4(res)) + res
        res = res.view(res.size(0), -1) # (batch_size, 32 * N**2)
        res_policy = nn.Softmax(dim=1)(self.fc_policy(res))
        res_value = nn.Tanh()(self.fc_value(res))
        return res_policy, res_value