import torch
from evaluator import evaluate
from net import ConvNet
from gameEngine import Actor, new_game, is_terminal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N_games = 1
N_rollouts_per_move = 100

new_net = ConvNet(device)
old_net = ConvNet(device)

print(evaluate(new_net, old_net, N_games, N_rollouts_per_move))
