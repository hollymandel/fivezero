from fivezero.gameEngine import State, legal_moves, step, Actor, new_game
from fivezero.net import ConvNet
from typing import List, Optional
import numpy as np
from torch import nn
import torch

c = 1.414

class Node:
    def __init__(self, move: Optional[int] = None, actor: Optional[Actor] = None, game_state: Optional[State] = None, parent: Optional["Node"] = None, epsilon: float = 0.01):
        self.children: List[Node] = [] 
        if actor is None:
            # Actor = next to play. Actor.POSITIVE always starts (with empty board)
            self.actor = Actor.POSITIVE if parent is None else -parent.actor
        else:
            self.actor = actor
        self.parent = parent

        if game_state is not None and actor is not None:
            assert game_state.player == actor, "Game state player does not match actor"
        self.game_state: State = game_state if game_state is not None else new_game()
        self.move: int = move # edge move (lands you to self.game_state)
        self.visits = epsilon
        self.value = 0 # number of wins for games that passes through this node, from the perspective of the actor to play next (self.actor)
        self.Q = 0 # value tensor of moving landing here. No grad.
        self.P = 0 # policy tensor of move landing here. No grad.

    def fully_expand(self, policy_net: ConvNet | None, overwrite: bool = False):
        if len(self.children) > 0 and not overwrite:
            raise ValueError("Node already has children, set overwrite=True to overwrite")

        if policy_net is None:
            policy, value = torch.zeros(1, 25), torch.zeros(1, 25)
        else:
            policy, value = policy_net.forward(policy_net.encode(self.game_state))

        # softmax the policy
        policy = nn.Softmax(dim=1)(policy)

        for possible_move in legal_moves(self.game_state):
            new_child = Node(move=possible_move, actor=-self.actor, game_state=step(self.game_state, possible_move), parent=self)
            new_child.P = policy[0,possible_move].item()
            new_child.Q = value[0,possible_move].item()
            self.children.append(new_child)

    def fully_expanded(self):
        return len(self.children) == len(legal_moves(self.game_state))

    def select(self):
        previous_max = -np.inf
        best_child = None

        for child in self.children:
            selection_value = child.Q + c * child.P * np.sqrt(self.visits) / (1 + child.visits)
            if selection_value > previous_max:
                previous_max = selection_value
                best_child = child
        if best_child is None:
            raise ValueError("No best child found")
        return best_child, best_child.move

    def uct_select(self):
        previous_max = -np.inf
        best_child = None

        for child in self.children:
            if child.visits > 1001:
                import pdb
                pdb.set_trace()
            selection_value = child.value / (1 + child.visits) + c * np.sqrt(self.visits) / (1 + child.visits)
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
        return f"{-self.actor} played {self.move}"
