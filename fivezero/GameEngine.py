"""
5x5 board, 4 in a row to win.

State:
- board: np.ndarray of shape (5,5), values in {-1,0,+1}
- player: int, +1 (X) or -1 (O) to move

Functions:
- new_game: create a new game state
- legal_mask: get the legal moves for a state
- step: apply a move to a state
"""

from dataclasses import dataclass
import numpy as np
from enum import IntEnum

N = 5
K = 4  # win length

class Actor(IntEnum):
    POSITIVE = 1
    NEGATIVE = -1

class State:
    def __init__(self, board: np.ndarray, player: Actor | None = None):
        self.board = board
        self.player = player if player is not None else Actor.POSITIVE # next-to-play
        
        if all(board.reshape(-1) == 0):
            assert self.player == Actor.POSITIVE
        
class Move:
    def __init__(self, index: int):
        self.index = index
        self.r, self.c = divmod(index, N)

    def __repr__(self):
        return f"Move({self.r}, {self.c})"

def new_game():
    return State(board=np.zeros((N, N), dtype=np.int8), player=1)

def legal_moves(s: State):
    mask = (s.board.reshape(-1) == 0)
    indices = np.where(mask)[0]
    return indices

def step(s: State, a: int):
    r, c = divmod(a, N)
    b = s.board.copy()
    b[r,c] = s.player # game state player is next-to-play for game state board
    return State(board=b, player=-s.player)

def winner(board: np.ndarray) -> int:
    # returns +1, -1, or 0
    dirs = [(0,1), (1,0), (1,1), (1,-1)]
    for r in range(N):
        for c in range(N):
            v = int(board[r, c])
            if v == 0: 
                continue
            for dr, dc in dirs:
                rr, cc = r + (K-1)*dr, c + (K-1)*dc
                if 0 <= rr < N and 0 <= cc < N:
                    if all(int(board[r+i*dr, c+i*dc]) == v for i in range(K)):
                        return v
    return 0

def terminal_value(s: State):
    """
    value from the perspective of the player to move in state s:
      -1: you have already lost (opponent just made a line)
       0: draw
    None: not terminal
    """
    w = winner(s.board)
    if w != 0:
        return -1  # if someone has a line, it's the previous mover, i.e. you lose
    if not np.any(s.board == 0):
        return 0
    return None

def is_terminal(s: State):
    return terminal_value(s) is not None

def canonical_board(s: State):
    # make "player to move" always be +1
    return (s.board * s.player).astype(np.int8)

def encode(s: State):
    # 2 planes: current stones, opponent stones (from current player's view)
    b = canonical_board(s)
    cur = (b == 1).astype(np.float32)
    opp = (b == -1).astype(np.float32)
    return np.stack([cur, opp], axis=0)

def render(s: State):
    sym = {1: "X", -1: "O", 0: "."}
    rows = [" ".join(sym[int(v)] for v in s.board[r]) for r in range(N)]
    turn = "X" if s.player == 1 else "O"
    return f"to move: {turn}\n" + "\n".join(rows)

def random_play(s: State):
    moves = legal_moves(s)
    if len(moves) == 0:
        raise ValueError("No legal moves left")
    return step(s, np.random.choice(moves))