import torch
from GameEngine import N

class ConvNet(torch.nn.Module):
    def __init__(self):
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
        self.fc_policy = torch.nn.Linear(32 * N**2, N**2)
        self.fc_value = torch.nn.Linear(32 * N**2, N**2)

    def forward(self, x):
        # skip connections at each layer after first
        res = self.relu1(self.conv1(x))
        res = self.relu1(self.conv2(res)) + res
        res = self.relu1(self.conv3(res)) + res
        res = self.relu1(self.conv4(res)) + res
        res = res.view(res.size(0), -1) # (batch_size, 32 * N**2)
        res_policy = self.fc_policy(res)
        res_value = self.fc_value(res)
        return res_policy, res_value