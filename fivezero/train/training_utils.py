import torch
import numpy as np
from fivezero.gameEngine import N
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simple_partition(n, k):
    """Partition n into k parts, where the parts are as equal as possible."""
    q, r = divmod(n, k)
    return [q + 1] * r + [q] * (k - r)

class TrainingBuffer:
    def __init__(self, epochs_in_buffer):
        self.epochs_in_buffer = epochs_in_buffer
        self.buffer = []
        self.game_index_buffer = []
        self.latest_epoch_index = 0

    def add_epoch(self, traces, game_index):
        self.buffer.append(traces)
        self.game_index_buffer.append(game_index)
        self.latest_epoch_index += 1

        if len(self.buffer) > self.epochs_in_buffer:
            self.buffer.pop(0)
            self.game_index_buffer.pop(0)

    def sample_from_buffer(self, batch_size):
        # randomly allocate batch size among all epochs in buffer
        batch_sizes = simple_partition(batch_size, len(self.buffer))
        
        # subsample each epoch in buffer by the batch size
        output_sample = []
        output_game_index = []
        output_epochs = []

        for i in range(len(self.buffer)):
            if batch_sizes[i] > len(self.buffer[i]):
                # use whole epoch if desired batch size is greater than epoch length
                output_sample.extend(self.buffer[i])
                output_game_index.extend(self.game_index_buffer[i])
                output_epochs.extend([ self.latest_epoch_index + (i-2) ] * len(self.buffer[i]))
            else:
                sample_indices = np.random.choice(len(self.buffer[i]), size=batch_sizes[i], replace=False)
                output_sample.extend([ self.buffer[i][j] for j in sample_indices ])
                output_game_index.extend([ self.game_index_buffer[i][j] for j in sample_indices ])
                output_epochs.extend([ self.latest_epoch_index + (i-2) ] * batch_sizes[i])

        return output_sample, output_game_index, output_epochs

def node_to_child_distribution(parent, temperature) -> np.ndarray:
    """cannonical representation of child distribution"""

    assert parent.fully_expanded(), "node_to_child_distribution called on non-fully-expanded node"
    assert len(parent.children) > 0, "node_to_child_distribution called on node with no children"
    distribution_tensor = np.zeros(N**2)
    for child in parent.children:
        distribution_tensor[child.move] = np.power(child.visits, 1/temperature)
    return torch.tensor(distribution_tensor / np.sum(distribution_tensor))

def training_step(batch_traces, net, value_criterion, policy_criterion, optimizer):

    batch_states = [ parent_node.game_state for parent_node, _, _ in batch_traces ]
    batch_states = torch.concatenate([ net.encode(state) for state in batch_states ], dim=0)

    batch_moves = torch.tensor([ child_node.move for _, child_node, _ in batch_traces ], dtype=torch.int64, device=device) 
    batch_zs = torch.tensor([ z * parent_node.actor for parent_node, _, z in batch_traces ], dtype=torch.float32, device=device)   
    child_values = [ parent_node.value_of_children(net) for parent_node, _, _ in batch_traces ]
    child_values = torch.tensor(child_values, dtype=torch.float32, device=device)

    
    # net predictions
    policy_predictions, value_predictions = net.forward(batch_states)
    value_predictions = [
        value_predictions[i,0] for i, _ in enumerate(batch_traces)
    ]
    value_predictions = torch.stack(value_predictions, dim=0)

    # as a sanity check, check the slope between value and predictions
    from scipy.stats import linregress
    # import pdb; pdb.set_trace()
    slope, intercept, r_value, p_value, std_err = linregress(child_values.detach().cpu().numpy().ravel(), policy_predictions.detach().cpu().numpy().ravel())
    # print(f"Slope: {slope}, Intercept: {intercept}, R-value: {r_value}, P-value: {p_value}, Std-err: {std_err}")


    # empirical distribution of child nodes
    empirical_policies = [
        node_to_child_distribution(parent_node, 1.0) for parent_node, _, _ in batch_traces
    ]
    empirical_policies = torch.stack(empirical_policies, dim=0)

    # basic sanity checks before computing loss
    assert torch.isfinite(empirical_policies).all(), "Non-finite values in empirical policies"
    assert (empirical_policies.sum(dim=1) > 0).all(), "Zero empirical policy row"

    value_loss = value_criterion(value_predictions, batch_zs)
    policy_loss = policy_criterion(policy_predictions, empirical_policies)
    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return value_loss.item(), policy_loss.item(), loss.item(), policy_predictions.detach().cpu().numpy(), empirical_policies.detach().cpu().numpy(), value_predictions.detach().cpu().numpy(), batch_zs.detach().cpu().numpy(), slope
