from fivezero.gameEngine import State, legal_moves, step, Actor, new_game
from fivezero.net import ConvNet
from typing import List, Optional
import numpy as np
from torch import nn
import torch
from copy import deepcopy

c = 10 * 1.414 # policy weighting factor
eta = 0.1 # exploration bonus

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
        self.move: int = move # incoming edge move (lands you to self.game_state)
        self.visits = 0
        self.value = 0 # number of wins for games that passes through this node, from the perspective of self.actor
                       # We are not backing up values because all games terminate. 

        # ! Caution Here ! Interpretation differes from the above which is primarily in terms of previous player
        # self.Q = 0 # value tensor of this state from the perspective of the actor to play next. In a sense
        #            # it "lags behind" the policy network by a move.
        self._P = 0 # policy tensor of LANDING HERE from the perspective of the PREVIOUS ACTOR. Used 
                   # solely by the parent for selection

        self.internal_selection_log = [] # debugging

    def puct_selection_value(self):
        # UCT selection value for THIS NODE from the perspective of the PREVIOUS ACTOR (parent.actor)
        return (
            -1 * self.value / (1 + self.visits) +
            c * ( self._P + eta ) * # exploration bonus 
            np.sqrt(self.parent.visits) / (1 + self.visits)
        )

    def fully_expand(self, policy_net: ConvNet | None, overwrite: bool = False):
        if len(self.children) > 0 and not overwrite:
            raise ValueError("Node already has children, set overwrite=True to overwrite")

        if policy_net is None:
            policy, value = torch.zeros(1, 25), torch.zeros(1, 1)
        else:
            policy, value = policy_net.forward(policy_net.encode(self.game_state))

        # softmax the policy
        policy = nn.Softmax(dim=1)(policy)

        for possible_move in legal_moves(self.game_state):
            new_child = Node(move=possible_move, actor=-self.actor, game_state=step(self.game_state, possible_move), parent=self)
            new_child._P = policy[0,possible_move].item()
            # issue here--need to do another network call with the child's game state to get the value
            # new_child.Q = value.item() # not currently used during play because we are not backing up values because all games terminate.
            self.children.append(new_child)

    def fully_expanded(self):
        return len(self.children) == len(legal_moves(self.game_state))

    def select(self):
        previous_max = -np.inf
        best_child = None

        all_selection_values = []
        for child in self.children:
            # need to flip the sign of the child value because it is from the perspective of the child
            selection_value = child.puct_selection_value()
            all_selection_values.append((child.move, deepcopy(selection_value)))
            if selection_value > previous_max:
                previous_max = selection_value
                best_child = child
        if best_child is None:
            raise ValueError("No best child found")
        else:
            pass
            # print(f"Selected child {best_child.move} with selection value {previous_max}  {[ int(x*100) for x in all_selection_values ]}")
        
        self.internal_selection_log.append((best_child.move, all_selection_values))
        return best_child, best_child.move

    def uct_select(self):
        previous_max = -np.inf
        best_child = None

        for child in self.children:
            # need to flip the sign of the child value because it is from the perspective of the child
            selection_value = -1 * child.value / (1 + child.visits) + c * np.sqrt(self.visits) / (1 + child.visits)
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
        # print the action that got us to this. A little weird since 
        # node should be thought of in terms of the next player, but
        # the next player's move is unknown so there's not much to report
        return f"{-self.actor} played {self.move}"

    def print_children(self):
    # print visit counts, values, and policies for each child in a nice table
        print(f"Children of {self}:")
        print(f"{'Move':<10} {'Visits':>8} {'Value':>8} {'Policy':>10} {'UCT Value':>10}")
        print("-" * 40)
        for child in self.children:
            print(
                f"{child.move:<10} "
                f"{child.visits:>8d} "
                f"{child.value:>8.1f} "
                f"{child._P:>10.3f} "
                f"{child.puct_selection_value():>10.3f}"
            )