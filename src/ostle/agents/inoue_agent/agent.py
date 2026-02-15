import math
import time
from dataclasses import dataclass

from ostle.agents.base import Agent
from ostle.core.board import (Cell, Move, applied_move, get_legal_moves,
                              is_winner)


@dataclass(frozen=True)
class SearchConfig:
    max_depth: int = 4
    time_buffer: float = 0.02


class InoueAgent(Agent):
    def __init__(self, config: SearchConfig | None = None):
        super().__init__(name="InoueAgent")
        self.config = config or SearchConfig()

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

        start_time = time.time()
        time_limit = max(0.0, time_remaining - self.config.time_buffer)

        best_move = legal_moves[0]
        best_score = -math.inf

        for depth in range(1, self.config.max_depth + 1):
            if self._is_time_over(start_time, time_limit):
                break
            score, move = self._search_root(
                board,
                player,
                depth,
                start_time,
                time_limit,
            )
            if move is not None:
                best_score = score
                best_move = move

        return best_move

    def _search_root(
        self,
        board: list[Cell],
        player: Cell,
        depth: int,
        start_time: float,
        time_limit: float,
    ) -> tuple[float, Move | None]:
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        moves = get_legal_moves(board, player)
        for move in moves:
            if self._is_time_over(start_time, time_limit):
                break
            next_board = applied_move(board, move)
            score = self._alphabeta(
                next_board,
                player.opponent(),
                depth - 1,
                alpha,
                beta,
                player,
                start_time,
                time_limit,
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_score, best_move

    def _alphabeta(
        self,
        board: list[Cell],
        turn: Cell,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Cell,
        start_time: float,
        time_limit: float,
    ) -> float:
        if self._is_time_over(start_time, time_limit):
            return self._evaluate(board, root_player)

        if depth == 0 or is_winner(board, root_player) or is_winner(
            board, root_player.opponent()
        ):
            return self._evaluate(board, root_player)

        moves = get_legal_moves(board, turn)
        if not moves:
            return self._evaluate(board, root_player)

        if turn == root_player:
            value = -math.inf
            for move in moves:
                if self._is_time_over(start_time, time_limit):
                    break
                next_board = applied_move(board, move)
                value = max(
                    value,
                    self._alphabeta(
                        next_board,
                        turn.opponent(),
                        depth - 1,
                        alpha,
                        beta,
                        root_player,
                        start_time,
                        time_limit,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = math.inf
        for move in moves:
            if self._is_time_over(start_time, time_limit):
                break
            next_board = applied_move(board, move)
            value = min(
                value,
                self._alphabeta(
                    next_board,
                    turn.opponent(),
                    depth - 1,
                    alpha,
                    beta,
                    root_player,
                    start_time,
                    time_limit,
                ),
            )
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _evaluate(self, board: list[Cell], player: Cell) -> float:
        if is_winner(board, player):
            return 10_000.0
        if is_winner(board, player.opponent()):
            return -10_000.0

        my_count = sum(1 for cell in board if cell == player)
        opp_count = sum(1 for cell in board if cell == player.opponent())
        material = (my_count - opp_count) * 100.0

        my_moves = len(get_legal_moves(board, player))
        opp_moves = len(get_legal_moves(board, player.opponent()))
        mobility = (my_moves - opp_moves) * 3.0

        return material + mobility

    def _is_time_over(self, start_time: float, time_limit: float) -> bool:
        if time_limit <= 0:
            return False
        return (time.time() - start_time) >= time_limit
