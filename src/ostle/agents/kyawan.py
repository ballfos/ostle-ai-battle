"""
基本のAlphaBetaエージェント
- 評価関数は勝敗のみ
- 探索深さは固定
"""

import random

from ostle.agents.base import Agent
from ostle.core.board import (
    Cell,
    Move,
    applied_move,
    get_legal_moves,
    is_same_board,
    is_winner,
)

WIN_SCORE = 10000
LOSE_SCORE = -20000


class KyawanAgent(Agent):
    def __init__(self):
        super().__init__(name="Kyawan")

    def calc_best_move(
        self,
        board,
        prev_board,
        player,
        time_remaining,
    ):
        return self.negamax(
            player,
            board,
            prev_board,
            depth=5,
            alpha=float("-inf"),
            beta=float("inf"),
        )[1]

    def negamax(
        self,
        player: Cell,
        board: list[Cell],
        prev_board: list[Cell],
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, Move | None]:
        if depth == 0:
            return self.evaluate(board, player, depth), None

        next_player = player.opponent()

        best_score = float("-inf")
        best_move = None

        # 序盤が単調になるので候補をランダムにシャッフルする
        moves = get_legal_moves(board, player)
        random.shuffle(moves)
        for move in moves:
            next_board = applied_move(board, move)
            if is_same_board(next_board, prev_board):
                continue  # 千日手回避
            score, _ = self.negamax(
                next_player,
                next_board,
                board[:],
                depth - 1,
                -beta,
                -alpha,
            )
            score = -score

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Bカット

        return best_score, best_move

    def evaluate(self, board: list[Cell], player: Cell, depth: int) -> float:
        if is_winner(board, player):
            return WIN_SCORE + depth  # 速く勝つほど高評価
        elif is_winner(board, player.opponent()):
            return LOSE_SCORE - depth  # 遅く負けるほど高評価
        else:
            return 0
