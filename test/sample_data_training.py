import pickle
from tkinter import N
import torch
from fivezero.train.update import training_step
from fivezero.net import ConvNet
from torch import nn
import torch.optim as optim
import random

random.seed(42)

with open("test/train/sample_training_data/traces.pkl", "rb") as f:
    traces = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ConvNet(device)
value_criterion = nn.MSELoss()
policy_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 50
BATCH_SIZE = 64


losses = []
value_losses = []
policy_losses = []
policy_predictions_list = []
empirical_policy_list = []
value_predictions_list = []
batch_zs_list = []
for i in range(EPOCHS):

    random.shuffle(traces)
    for batch in range(0, len(traces), BATCH_SIZE):
        batch_traces = traces[batch:batch+BATCH_SIZE]
        value_loss, policy_loss, loss, policy_predictions, empirical_policies, value_predictions, batch_zs = training_step(batch_traces, net, value_criterion, policy_criterion, optimizer)

        losses.append(loss)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
        policy_predictions_list.append(policy_predictions)
        empirical_policy_list.append(empirical_policies)
        value_predictions_list.extend(value_predictions)
        batch_zs_list.extend(batch_zs)
        # print("Value loss: ", value_loss)
        # print("Policy loss: ", policy_loss)
        # print("Loss: ", loss)

import pickle
save_data = (losses, value_losses, policy_losses, policy_predictions_list, empirical_policy_list, value_predictions_list, batch_zs_list)
with open("/Users/hollymandel/Documents/FiveZero/test/train/sample_training_data/training_data.pkl", "wb") as f:
    pickle.dump(save_data, f)
# with open("test/train/sample_data_training/losses.pkl", "wb") as f:
#     pickle.dump(losses, f)
# with open("test/train/sample_data_training/value_losses.pkl", "wb") as f:
#     pickle.dump(value_losses, f)
# with open("test/train/sample_data_training/policy_losses.pkl", "wb") as f:
#     pickle.dump(policy_losses, f)