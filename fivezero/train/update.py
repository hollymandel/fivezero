import torch
import numpy as np
from fivezero.gameEngine import N

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # net predictions
    policy_predictions, value_predictions = net.forward(batch_states)
    value_predictions = [
        value_predictions[i, child_node.move] for i, (_, child_node, _) in enumerate(batch_traces)
    ]
    value_predictions = torch.stack(value_predictions, dim=0)

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

    return value_loss.item(), policy_loss.item(), loss.item(), policy_predictions.detach().cpu().numpy(), empirical_policies.detach().cpu().numpy(), value_predictions.detach().cpu().numpy(), batch_zs.detach().cpu().numpy()
