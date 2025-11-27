from GameEngine import state
from typing import List
import numpy as np
import random

class Node:
    def __init__(self, move: int = None, actor: int = None, game_state: state = None, parent: "Node" = None, epsilon: float =0.01):
        self.children: List[Node] = []
        self.actor = actor
        self.parent = parent
        self.game_state: state = game_state # board after move
        self.move: int = move # edge move (lands you to board)
        self.visits = epsilon

    # def max_child(self, use_randomness = False, C = C):
    #     max_utc = -1
    #     max_child = None
    #     for child in self.children:
    #         if use_randomness:
    #             random_add = np.random.randn()
    #         else:
    #             random_add = 0
    #         utc = child.total_return / child.visits + C* np.sqrt(np.log(self.visits) / child.visits + 1e-3) + random_add
    #         if child.visits == epsilon:
    #             utc = 1e10
    #         if max_child is None or utc >= max_utc:
    #             max_child = child
    #             max_utc = utc
    #     return max_child

    def __repr__(self):
        return f"{self.actor} plays {self.move}"