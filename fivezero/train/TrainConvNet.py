import pdb

from fivezero.gameEngine import *
from fivezero.net import ConvNet
from fivezero.tree import Node
from fivezero.train.step import play_step
from fivezero.train.update import training_step
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import pickle

N_epochs = 10
games_per_epoch = 10
mcts_rollouts_per_move = 1000
batch_size = 64
N_training_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root = Node(move=None, actor=1, game_state=new_game())
net = ConvNet(device)
value_criterion = nn.MSELoss()
policy_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(N_epochs):
    traces = []
    wins_p = 0
    wins_n = 0
    draws = 0

    for game in range(games_per_epoch):
        game_start = time.time()
        # play game, collect rollouts
        trace = []
        game_state = new_game()
        parent_node = root
        while not is_terminal(game_state):
            parent_node, child_node = play_step(parent_node, net, net, temperature=1.0, N_rollouts_per_move=mcts_rollouts_per_move)
            trace.append((parent_node, child_node))
            game_state = child_node.game_state
            parent_node = child_node

        z = winner(game_state.board)
        if z == 1:
            wins_p += 1
        elif z == -1:
            wins_n += 1
        elif z == 0:
            draws += 1
        trace = [(parent_node, child_node, z) for parent_node, child_node in trace]

        traces.extend(trace)
        game_end = time.time()
        print(f"Game {game} of epoch {epoch} took {game_end - game_start} seconds")

    assert len(traces) > 0, f"No traces collected in epoch {epoch}"

    import pdb; pdb.set_trace()
    # update in batches
    for training_epoch in range(N_training_epochs):
        random.shuffle(traces)
        for batch in range(0, len(traces), batch_size):
            
            batch_traces = traces[batch:batch+batch_size]
            value_loss, policy_loss, loss, policy_predictions, empirical_policies, value_predictions, batch_zs = training_step(batch_traces, net, value_criterion, policy_criterion, optimizer)

            print("Value loss: ", value_loss)
            print("Policy loss: ", policy_loss)
            print("Loss: ", loss)

            # batch_states = [ parent_node.game_state for parent_node, _, _ in batch_traces ]
            # batch_states = torch.concatenate([ net.encode(state) for state in batch_states ], dim=0)

            # batch_moves = torch.tensor([ child_node.move for _, child_node, _ in batch_traces ], dtype=torch.int64, device=device) 
            # batch_zs = torch.tensor([ z for _, _, z in batch_traces ], dtype=torch.float32, device=device)   

            # # net predictions
            # policy_predictions, value_predictions = net.forward(batch_states)
            # value_predictions = [
            #     value_predictions[0, child_node.move] for i, (_, child_node, _) in enumerate(batch_traces)
            # ]
            # value_predictions = torch.stack(value_predictions, dim=0)

            # # empirical distribution of child nodes
            # empirical_policies = [
            #     node_to_child_distribution(parent_node, 1.0) for parent_node, _, _ in batch_traces
            # ]
            # empirical_policies = torch.stack(empirical_policies, dim=0)

            # # basic sanity checks before computing loss
            # assert torch.isfinite(empirical_policies).all(), "Non-finite values in empirical policies"
            # assert (empirical_policies.sum(dim=1) > 0).all(), "Zero empirical policy row"

            # value_loss = value_criterion(value_predictions, batch_zs)
            # policy_loss = policy_criterion(policy_predictions, empirical_policies)
            # loss = value_loss + policy_loss

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # save some data 
            # save_tuple = (value_loss.item(), policy_loss.item(), loss.item(), policy_predictions.detach().cpu().numpy(), empirical_policies.detach().cpu().numpy(), value_predictions.detach().cpu().numpy(), batch_zs.detach().cpu().numpy())
            # with open(f"/Users/hollymandel/Documents/FiveZero/fivezero/train/dec_11/training_data_{epoch}_{training_epoch}.pkl", "wb") as f:
            #     pickle.dump(save_tuple, f)

            # print("Loss: ", loss.item())

    # evaluator -- is updated model better? skip for now


print(f"Wins for positive player: {wins_p}")
print(f"Wins for negative player: {wins_n}")
print(f"Draws: {draws}")
        


        
