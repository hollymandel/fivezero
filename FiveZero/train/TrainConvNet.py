from fivezero.gameEngine import *
from fivezero.net import ConvNet
from fivezero.tree import Node
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

N_epochs = 1
games_per_epoch = 10
mcts_rollouts_per_move = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root = Node(move=None, actor=1, game_state=new_game())
net = ConvNet(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def mcts_rollout(node: Node):
    """
    Perform MCTS rollouts from the given (root) node. 
    Returns the value of the root node only.
    """
    raise NotImplementedError("MCTS rollouts not implemented")

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

    # train the network

assert False

            
