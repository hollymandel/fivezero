from tree import Node
from net import ConvNet
from dataclasses import dataclass
from gameEngine import State, new_game, step, random_play, Move, is_terminal, terminal_value, Actor
from typing import List
import pdb
@dataclass
class trace_sample:
    game_state: State
    policy: dict[Move, float]
    value: float

def mcts_rollout(root: Node, netp: ConvNet, netn: ConvNet | None = None) -> None:
    """
    Perform a single MCTS rollout from the given (root) node. Root node 
    may be a "blank" tree or populated. Node is modified in place.
    Trace is not recorded; rather, rollout value and frequencies are backpropped
    into the node and its children.
    """
    netp.eval() # net for positive player
    if netn is not None: # net for negative player
        netn.eval()
    else:
        netn = netp

    game_state = root.game_state
    node = root

    while not is_terminal(game_state):
        use_net = netp if node.actor == 1 else netn
        if node.fully_expanded(): 
            # selection
            node, action = node.select()
            game_state = step(game_state, action)
            # print(f"Selected action: {action}")
            # print(f"Game state: \n{game_state.board}")
            # print(f"Node: {node.game_state.board}")

        else:
            # expansion
            node.fully_expand(netp)
            node, action = node.select()
            game_state = step(game_state, action)
            # print(f"Expanded action: {action}")
            # print(f"Game state: \n{game_state.board}")
            # print(f"Node: {node.game_state.board}")
            break


    # print("moved on to random play\n\n")
    while not is_terminal(game_state):
        # random play
        game_state = random_play(game_state)

    # backpropagate the rollout value and frequencies into the node and its children
    value = terminal_value(game_state)
    if value is None:
        raise ValueError("Game state is not terminal")
    
    while node.parent is not None:
        node = node.parent
        node.value += value * node.actor
        node.visits += 1
    node.value += value * node.actor
    node.visits += 1
