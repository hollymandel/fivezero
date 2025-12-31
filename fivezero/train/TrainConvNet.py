from fivezero.gameEngine import *
from fivezero.net import ConvNet
from fivezero.tree import Node
from fivezero.train.step import play_step
from fivezero.train.training_utils import training_step, TrainingBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle

N_epochs = 1000
games_per_epoch = 10
epochs_in_buffer = 3
mcts_rollouts_per_move = 256
batch_size = 64
training_batches_per_epoch = 20
use_checkpoint = None
# use_checkpoint = "/Users/hollymandel/Documents/FiveZero/data/dec_28/latest_model_2.pth"
temperature_decay = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load model if it exists
net = ConvNet(device)

if use_checkpoint is not None:
    print(f"Loading checkpoint from {use_checkpoint}")
    net.load_state_dict(torch.load(use_checkpoint))
else:
    print("No checkpoint provided, starting from scratch")

value_criterion = nn.MSELoss()
policy_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

data_dictionary = {
    "losses": [],
    "value_losses": [],
    "policy_losses": [],
    "policy_predictions_list": [],
    "empirical_policy_list": [],
    "value_predictions_list": [],
    "batch_zs_list": [],
    "epoch_index": [],
    "batch_index": [],
    "game_index": [],
    "slope_list": []
}
training_buffer = TrainingBuffer(epochs_in_buffer)

for epoch in range(N_epochs):
    traces = []
    wins_p = 0
    wins_n = 0
    draws = 0

    epoch_traces = []
    game_index_traces = []
    for game in range(games_per_epoch):
        game_start = time.time()
        # play game, collect rollouts
        trace = []
        game_state = new_game()
        
        root = Node(move=None, actor=1, game_state=new_game())
        parent_node = root

        while not is_terminal(game_state):
            parent_node, child_node = play_step(parent_node, net, net, temperature=temperature_decay**epoch, N_rollouts_per_move=mcts_rollouts_per_move)
            trace.append((parent_node, child_node))
            game_state = child_node.game_state

            # avoiding MCTS contamination between steps. This is necessary to preserve the statistics
            # for policy training. If you continue expanding from the same root, the exploration distribution
            # will heavily skew towards the node you selected and not be useful for training the policy.
            # However, the trace should keep pointers to the nodes used at the time. A check is whether
            # the parent nodes contain MCTS_ROLLOUTS_PER_MOVE total visits.
            parent_node = Node(move=None, actor=child_node.actor, game_state=game_state, parent=None)

        z = asymmetric_winner(game_state.board)
        if z == 1:
            wins_p += 1
        elif z == -1:
            wins_n += 1
        else:
            draws += 1
        trace = [(parent_node, child_node, z) for parent_node, child_node in trace]
        epoch_traces.extend(trace)
        game_index_traces.extend([ game ] * len(trace))

        game_end = time.time()
        # print(f"Game {game} of epoch {epoch} took {game_end - game_start} seconds")

    # print("Wins positive: ", wins_p)
    # print("Wins negative: ", wins_n)
    # print("Draws: ", draws)

    training_buffer.add_epoch(epoch_traces, game_index_traces)

        
    # update in batches
    for batch in range(training_batches_per_epoch):
        average_value_loss = 0
        average_policy_loss = 0
        average_loss = 0
        average_slope = 0

        batch_traces, batch_game_indices, batch_epochs = training_buffer.sample_from_buffer(batch_size)
        value_loss, policy_loss, loss, policy_predictions, empirical_policies, value_predictions, batch_zs, slope = training_step(batch_traces, net, value_criterion, policy_criterion, optimizer)

        # fudged for printing niceness
        average_value_loss += value_loss# / training_batches_per_epoch
        average_policy_loss += policy_loss# / training_batches_per_epoch
        average_loss += loss# / training_batches_per_epoch
        average_slope += slope# / training_batches_per_epoch

        data_dictionary["epoch_index"].append(batch_epochs)
        data_dictionary["game_index"].append(batch_game_indices)
        data_dictionary["losses"].append(loss)
        data_dictionary["value_losses"].append(value_loss)
        data_dictionary["policy_losses"].append(policy_loss)
        data_dictionary["policy_predictions_list"].append(policy_predictions)
        data_dictionary["empirical_policy_list"].append(empirical_policies)
        data_dictionary["value_predictions_list"].append(value_predictions)
        data_dictionary["batch_zs_list"].append(batch_zs)
        data_dictionary["slope_list"].append(slope)

    # print value, policy, and total loss in one line
    print(f"Epoch {epoch} of {N_epochs} complete. Average value loss: {average_value_loss:.2f}, average policy loss: {average_policy_loss:.2f}, average total loss: {average_loss:.2f}, average slope: {average_slope:.2f}")

    with open(f"/Users/hollymandel/Documents/FiveZero/data/dec_28/all_data.pkl", "wb") as f:
        pickle.dump(data_dictionary, f)

    # checkpoint of model
    torch.save(net.state_dict(), f"/Users/hollymandel/Documents/FiveZero/data/dec_28/latest_model.pth")



        
