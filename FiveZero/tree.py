from FiveZero.GameEngine import State
from typing import List
import numpy as np
from FiveZero.GameEngine import legal_moves, step
from FiveZero.net import ConvNet
import numpy as np

c = 1.414

class Node:
    def __init__(self, move: int = None, actor: int = None, game_state: State = None, parent: "Node" = None, epsilon: float =0.01):
        self.children: List[Node] = []
        self.actor = actor
        self.parent = parent
        self.game_state: State = game_state # current board state
        self.move: int = move # edge move (lands you to self.game_state)
        self.visits = epsilon

    def fully_expanded(self):
        return len(self.children) == len(legal_moves(self.game_state))

    def select(self, policy_net: ConvNet):
        previous_max = 0
        best_child = None
        policy, value = policy_net.forward([self.game_state]) 
        for child in self.children:
            possible_move = child.move
            selection_value = value[0,possible_move].item() + c * policy[0,possible_move].item() * np.sqrt(self.visits) / (1 + child.visits)
            if selection_value > previous_max:
                previous_max = selection_value
                best_child = child
        if best_child is None:
            raise ValueError("No best child found")
        return best_child, best_child.move

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