"""
ちょっと工夫したAlphaBetaエージェント
- 評価関数は勝敗+生存数+中央支配
- 探索深さは動的
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
SCORE_PER_PIECE = 100
SCORE_PER_CONTROL = 10
CENTER_POSITIONS = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]


class KyawanAgentV2(Agent):
    def __init__(self):
        super().__init__(name="KyawanV2")

    def calc_best_move(
        self,
        board,
        prev_board,
        player,
        time_remaining,
    ):
        if time_remaining < 1000:
            depth = 3
        elif time_remaining < 3000:
            depth = 4
        else:
            depth = 5

        return self.negamax(
            player,
            board,
            prev_board,
            depth=depth,
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
        opponent = player.opponent()
        if depth == 0:
            return self.evaluate(board, player, opponent, depth), None

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
                opponent,
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

    def evaluate(
        self, board: list[Cell], player: Cell, opponent: Cell, depth: int
    ) -> float:
        # 勝敗の評価
        if is_winner(board, player):
            return WIN_SCORE + depth  # 速く勝つほど高評価
        elif is_winner(board, opponent):
            return LOSE_SCORE - depth  # 遅く負けるほど高評価

        score = 0

        # 生存数の評価
        player_count = sum(1 for cell in board if cell == player)
        opponent_count = sum(1 for cell in board if cell == opponent)
        score += (player_count - opponent_count) * SCORE_PER_PIECE

        # 中央支配の評価
        # center_positions = CENTER_POSITIONS
        # for x, y in center_positions:
        #     index = xy_to_index(x, y)
        #     if board[index] == player:
        #         score += SCORE_PER_CONTROL
        #     elif board[index] == opponent:
        #         score -= SCORE_PER_CONTROL

        return score
