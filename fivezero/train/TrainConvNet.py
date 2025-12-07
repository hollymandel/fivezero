from gameEngine import *
from net import ConvNet
from tree import Node
from step import play_step
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

N_epochs = 1
games_per_epoch = 10
mcts_rollouts_per_move = 20
batch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root = Node(move=None, actor=1, game_state=new_game())
net = ConvNet(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def represent_child_distribution(child_visits_raw: dict[int, int], temperature) -> np.ndarray:
    """cannonical representation of child distribution for 
    MCTS rollouts to match shape of policy network output"""

    distribution_tensor = torch.zeros(N**2)
    for move, visits in child_visits_raw.items():
        distribution_tensor[move] = np.power(visits, 1/temperature)
    return distribution_tensor / torch.sum(distribution_tensor)

for epoch in range(N_epochs):
    traces = []
    for game in range(games_per_epoch):
        # evaluator

        # play game, collect rollouts
        trace = []
        game_state = new_game()
        while not is_terminal(game_state):
            child, child_visits_raw = play_step(game_state, net, net, temperature=1.0, N_rollouts_per_move=mcts_rollouts_per_move)
            trace.append((game_state, represent_child_distribution(child_visits_raw, 1.0)))
            game_state = child.game_state

        z = winner(game_state.board)
        trace = [(state, move, z) for state, move in trace]
        trace = trace[::-1]

        traces.append(trace)
    
    # update in batches
    for batch in range(0, len(traces), batch_size):
        batch_traces = traces[batch:batch+batch_size]
        batch_states = [ state for trace in batch_traces for state, _, _ in trace ]
        batch_states = torch.stack([ net.encode(state) for state in batch_states ])

        batch_moves = [ move for trace in batch_traces for _, move, _ in trace ]
        batch_zs = [ z for trace in batch_traces for _, _, z in trace ]

        # net predictions
        policy_predictions, value_predictions = net.forward(batch_states)
        


        
