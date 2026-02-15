import time
from random import choice

from ostle.agents.base import Agent
from ostle.core.board import get_legal_moves


class RandomAgent(Agent):
    def __init__(self):
        super().__init__(name="RandomAgent")

    def calc_best_move(
        self,
        board,
        prev_board,
        player,
        time_remaining,
    ):
        legal_moves = get_legal_moves(board, player)
        if not legal_moves:
            raise ValueError("No legal moves available")

        time.sleep(1)

        return choice(legal_moves)
