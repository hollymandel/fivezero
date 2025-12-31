from dataclasses import dataclass
from typing import List

from fivezero.gameEngine import (
    Move,
    State,
    asymmetric_winner,
    is_terminal,
    random_play,
    step,
    winner,
)
from fivezero.net import ConvNet
from fivezero.tree import Node


def mcts_rollout(
    root: Node, netp: ConvNet | None, netn: ConvNet | None = None, use_uct: bool = False
) -> None:
    """
    Perform a single MCTS rollout from the given (root) node. Root node
    may be a "blank" tree or populated. Node is modified in place.
    Trace is not recorded; rather, rollout value and frequencies are backpropped
    into the node and its children.
    """
    if netp is None and use_uct is False:
        raise ValueError(
            "Must provide a network for positive player if use_uct is False"
        )

    if netp is not None:
        netp.eval()  # net for positive player
        if netn is not None:  # net for negative player
            netn.eval()
        else:
            netn = netp

    game_state = root.game_state
    node = root

    while not is_terminal(game_state):
        use_net = netp if node.actor == 1 else netn
        if node.fully_expanded():
            # selection
            if use_uct:
                node, action = node.uct_select()
            else:
                node, action = node.select()
            game_state = step(game_state, action)

        else:
            # expansion
            node.fully_expand(use_net)
            if use_uct:
                node, action = node.uct_select()
            else:
                node, action = node.select()
            game_state = step(game_state, action)
            break

    while not is_terminal(game_state):
        # random play
        game_state = random_play(game_state)

    # backpropagate the rollout value and frequencies into the node and its children
    # In principal you would backpropagate the value of the terminal state, but we don't because all games terminate.
    value = asymmetric_winner(game_state.board)

    if value is None:
        raise ValueError("Game state is not terminal")

    while node.parent is not None:
        node.value += value * node.actor
        node.visits += 1
        node = node.parent

    node.value += value * node.actor
    node.visits += 1
