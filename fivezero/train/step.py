from tree import Node
from mcts import mcts_rollout
from net import ConvNet
from gameEngine import Actor, new_game, is_terminal, winner, State
import numpy as np


def play_step(parent_node: Node, player_net: ConvNet, opponent_net: ConvNet, temperature: float = 1.0, N_rollouts_per_move: int = 100) -> float:
    """
    Evaluate the given network on the given root node.
    """

    start_state = parent_node.game_state

    if is_terminal(start_state):
        raise ValueError("Game is already terminal")

    # populate node with MCTS rollouts
    for _ in range(N_rollouts_per_move):
        if start_state.player == Actor.POSITIVE:
            mcts_rollout(parent_node, player_net, opponent_net)
        else:
            mcts_rollout(parent_node, opponent_net, player_net)

    # determine distribution of the root node's children
    child_visits_raw = { child.move: child.visits for child in parent_node.children }
    child_distribution = [ np.power(child.visits, 1/temperature) for child in parent_node.children ]
    child_distribution = child_distribution / np.sum(child_distribution)
    # select a child based on the distribution
    child_node = np.random.choice(parent_node.children, p=child_distribution)

    return parent_node, child_node


        
        
