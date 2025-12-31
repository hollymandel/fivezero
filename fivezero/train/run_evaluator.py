import numpy as np
import torch

from fivezero.gameEngine import Actor, is_terminal, new_game
from fivezero.net import ConvNet
from fivezero.train.evaluator import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N_games = 1
N_rollouts_per_move = 100

new_net = ConvNet(device)
old_net = ConvNet(device)

print(evaluate(new_net, old_net, N_games, N_rollouts_per_move))
