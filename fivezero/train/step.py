from fivezero.tree import Node
from fivezero.train.mcts import mcts_rollout
from fivezero.net import ConvNet
from fivezero.gameEngine import Actor, new_game, is_terminal, winner, State
import numpy as np


def step(start_state: State, player_net: ConvNet, opponent_net: ConvNet, temperature: float = 1.0, N_rollouts_per_move: int = 100) -> float:
    """
    Evaluate the given network on the given root node.
    """

    if is_terminal(start_state):
        raise ValueError("Game is already terminal")

    # create an MCTS node for the current game state
    node = Node(game_state=start_state, parent=None, actor=game_state.player, move=None)

    # populate node with MCTS rollouts
    for _ in range(N_rollouts_per_move):
        if start_state.player == Actor.POSITIVE:
            mcts_rollout(node, player_net, opponent_net)
        else:
            mcts_rollout(node, opponent_net, player_net)

    # determine distribution of the root node's children
    child_distribution = [ np.power(child.visits, 1/temperature) for child in node.children ]
    child_distribution = child_distribution / np.sum(child_distribution)
    # select a child based on the distribution
    child = np.random.choice(node.children, p=child_distribution)

    return child


        
        