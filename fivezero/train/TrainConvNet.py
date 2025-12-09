import pdb

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
games_per_epoch = 30
mcts_rollouts_per_move = 100
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root = Node(move=None, actor=1, game_state=new_game())
net = ConvNet(device)
value_criterion = nn.MSELoss()
policy_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def loss_function(policy_predictions, value_predictions, batch_moves, batch_zs):
    value_loss = value_criterion(value_predictions, batch_zs)
    policy_loss = policy_criterion(policy_predictions, batch_moves)
    return value_loss + policy_loss

# def represent_child_distribution(child_visits_raw: dict[int, int], temperature) -> np.ndarray:
#     """cannonical representation of child distribution for 
#     MCTS rollouts to match shape of policy network output"""

#     distribution_tensor = torch.zeros(N**2)
#     for move, visits in child_visits_raw.items():
#         distribution_tensor[move] = np.power(visits, 1/temperature)
#     return distribution_tensor / torch.sum(distribution_tensor)

def node_to_child_distribution(parent, temperature) -> np.ndarray:
    """cannonical representation of child distribution"""

    distribution_tensor = torch.zeros(N**2)
    for child in parent.children:
        distribution_tensor[child.move] = torch.power(child.visits, 1/temperature)
    return distribution_tensor / torch.sum(distribution_tensor)

for epoch in range(N_epochs):
    traces = []
    wins_p = 0
    wins_n = 0
    draws = 0

    for game in range(games_per_epoch):
        # evaluator

        # play game, collect rollouts
        trace = []
        game_state = new_game()
        while not is_terminal(game_state):
            child, _ = play_step(game_state, net, net, temperature=1.0, N_rollouts_per_move=mcts_rollouts_per_move)
            trace.append((game_state, child))
            game_state = child.game_state

        z = winner(game_state.board)
        if z == 1:
            wins_p += 1
        elif z == -1:
            wins_n += 1
        elif z == 0:
            draws += 1
        trace = [(state, child_node, z) for state, child_node in trace]
        trace = trace[::-1]

        traces.extend(trace)
    
    # update in batches
    for batch in range(0, len(traces), batch_size):
        batch_traces = traces[batch:batch+batch_size]
        batch_states = [ state for state, _, _ in batch_traces ]
        batch_states = torch.concatenate([ net.encode(state) for state in batch_states ], dim=0)

        batch_moves = torch.tensor([ child.move for _, child, _ in batch_traces ], dtype=torch.int64, device=device) 
        batch_values = torch.stack([ child.Q for  _, child, _ in batch_traces ])
        batch_zs = torch.tensor([ z for _, _, z in batch_traces ], dtype=torch.float32, device=device)   
        batch_policies = [
            node_to_child_distribution(node, 1.0) for _, node, _  in batch_traces
        ]
        batch_policies = torch.stack(batch_policies, dim=0)

        # # net predictions
        # policy_predictions, value_predictions = net.forward(batch_states)
        # # surely not? just compute V directly for parent state?
        # state_values = torch.sum(policy_predictions * value_predictions, dim=1) # (batch_size, N**2)
        import pdb
        pdb.set_trace()
        loss = loss_function(batch_policies, batch_values, batch_moves, batch_zs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pdb.set_trace()


print(f"Wins for positive player: {wins_p}")
print(f"Wins for negative player: {wins_n}")
print(f"Draws: {draws}")
        


        
