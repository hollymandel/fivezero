from gameEngine import State, legal_moves, step, Actor, new_game
from net import ConvNet
from typing import List, Optional
import numpy as np

c = 1.414

class Node:
    def __init__(self, move: Optional[int] = None, actor: Optional[Actor] = None, game_state: Optional[State] = None, parent: Optional["Node"] = None, epsilon: float = 0.01):
        self.children: List[Node] = []
        if actor is None:
            # Actor.POSITIVE always starts
            self.actor = Actor.POSITIVE if parent is None else -parent.actor
        else:
            self.actor = actor
        self.parent = parent
        self.game_state: State = game_state if game_state is not None else new_game()
        self.move: int = move # edge move (lands you to self.game_state)
        self.visits = epsilon
        self.value = 0
        self.Q = 0 # latest value estimate for this node. Retain grad.
        self.P = 0

    def fully_expand(self, policy_net: ConvNet, overwrite: bool = False):
        if len(self.children) > 0 and not overwrite:
            raise ValueError("Node already has children, set overwrite=True to overwrite")

        policy, value = policy_net.forward(policy_net.encode(self.game_state))

        for possible_move in legal_moves(self.game_state):
            new_child = Node(move=possible_move, actor=-self.actor, game_state=step(self.game_state, possible_move), parent=self)
            new_child.P = policy[0,possible_move]
            new_child.Q = value[0,possible_move]
            self.children.append(new_child)

    def fully_expanded(self):
        return len(self.children) == len(legal_moves(self.game_state))

    def select(self):
        previous_max = -np.inf
        best_child = None

        for child in self.children:
            selection_value = child.Q.item() + c * child.P.item() * np.sqrt(self.visits) / (1 + child.visits)
            if selection_value > previous_max:
                previous_max = selection_value
                best_child = child
        if best_child is None:
            raise ValueError("No best child found")
        return best_child, best_child.move

    # def select(self, policy_net: ConvNet):
    #     previous_max = -np.inf
    #     best_child = None
    #     state_encoding = policy_net.encode(self.game_state)
    #     moves_policy = policy_net.forward_policy(state_encoding)

    #     for child in self.children:
    #         possible_move = child.move
    #         conditional_value = policy_net.forward_value(policy_net.encode(child.game_state))
    #         selection_value = state_value.item() + c * moves_policy[0,possible_move].item() * np.sqrt(self.visits) / (1 + child.visits)
    #         if selection_value > previous_max:
    #             previous_max = selection_value
    #             best_child = child
    #     if best_child is None:
    #         raise ValueError("No best child found")
    #     return best_child, best_child.move

    def expand(self):
        child_moves = [ child.move for child in self.children ]
        for possible_move in legal_moves(self.game_state):
            if possible_move not in child_moves:
                new_child = Node(move=possible_move, actor=-self.actor, game_state=step(self.game_state, possible_move), parent=self)
                self.children.append(new_child)
                return new_child, possible_move
        raise ValueError("No legal moves left to expand")

    def __repr__(self):
        return f"{self.actor} plays {self.move}"
