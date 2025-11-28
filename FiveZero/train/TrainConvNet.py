from FiveZero.GameEngine import *
from FiveZero.net import ConvNet
from FiveZero.tree import Node
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

N_epochs = 1
games_per_epoch = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root = Node(move=None, actor=1, game_state=new_game())
net = ConvNet(device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(N_epochs):
    net.train()
    traces = []
    for i in range(games_per_epoch):
        trace = []
        game_state = new_game()
        node = root
        while terminal_value(game_state) is None:
            if node.fully_expanded(): 
                # selection
                node, action = node.select(net)
                game_state = step(game_state, action)
                trace.append((game_state, action))

            else:
                # expansion
                node, action = node.expand()
                game_state = step(game_state, action)
                trace.append((game_state, action))
                break

        while terminal_value(game_state) is None:
            # random play
            game_state = random_play(game_state)
            trace.append((game_state, None))
        traces.append(trace)

assert False

            
