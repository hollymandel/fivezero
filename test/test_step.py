# test train/step.py by setting up games with clear best choices for each player and checking that the step function chooses the best choice

from fivezero.train.step import play_step
from fivezero.gameEngine import new_game, is_terminal, winner
from fivezero.tree import Node
import numpy as np

N_rollouts_per_move = 512

def test_step():
    winners = []
    for _ in range(10):
        game = new_game()
        game.board = np.array([
            [1, 1, 1, 0, 0],
            [0, -1, 0, 0, 0],
            [0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0]
        ])
        game.player = 1
        root = Node(game_state=game, actor=1, move=None) # new tree for each game
        parent_node = root
        while not is_terminal(game):
            parent_node, child_node = play_step(parent_node, player_net=None, opponent_net=None, use_uct=True, N_rollouts_per_move=N_rollouts_per_move, temperature=0)
            game = child_node.game_state
            # detach parent node to prevent excessive backprop from later steps
            parent_node = Node(game_state=game, actor=child_node.actor, move=child_node.move, parent=None)

        this_winner = winner(game.board)
        winners.append(this_winner)



    assert sum(winner == 1 for winner in winners) == 10, winners

    assert np.abs(sum([ x.visits for x in root.children]) - N_rollouts_per_move) < 1, f"Sum of visits for root children is {sum(root.children.visits)}, expected {N_rollouts_per_move}"
