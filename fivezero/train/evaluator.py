from fivezero.tree import Node
from fivezero.train.mcts import mcts_rollout
from fivezero.net import ConvNet
from fivezero.gameEngine import Actor, new_game, is_terminal, winner
import numpy as np

TEMPERATURE = 1.0

def evaluate(new_net: ConvNet, old_net: ConvNet, N_games: int, N_rollouts_per_move: int) -> float:
    """
    Evaluate the given network on the given root node.
    """

    winners = []
    for game in range(N_games):
        if game % 2 == 0:
            new_player = Actor.POSITIVE
        else:
            new_player = Actor.NEGATIVE
            
        # Game Tree maintained independent of MCTS search for each step
        game_state = new_game()
        game_trace = []

        while not is_terminal(game_state):
            # create an MCTS node for the current game state
            node = Node(game_state=game_state, parent=None, actor=game_state.player, move=None)
            # populate node with MCTS rollouts
            for _ in range(N_rollouts_per_move):
                if new_player == Actor.POSITIVE:
                    mcts_rollout(node, new_net, old_net)
                else:
                    mcts_rollout(node, old_net, new_net)        
            # determine distribution of the root node's children
            child_distribution = [ np.power(child.visits, 1/TEMPERATURE) for child in node.children ]
            child_distribution = child_distribution / np.sum(child_distribution)
            # select a child based on the distribution
            child = np.random.choice(node.children, p=child_distribution)
            # add the child to the game trace
            game_trace.append(child)
            # update the game state
            game_state = child.game_state

        winners.append(winner(game_state.board))

    return np.mean(winners)
    


        
        