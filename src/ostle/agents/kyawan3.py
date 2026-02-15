"""
さらに工夫したAlphaBetaエージェント
- 評価関数は勝敗+生存数+中央支配
- 探索深さは動的
- Move候補に優先順位をつける
"""

from ostle.agents.base import Agent
from ostle.core.board import (
    Cell,
    Move,
    applied_move,
    get_legal_moves,
    is_in_board,
    is_same_board,
    is_winner,
    xy_to_index,
)

WIN_SCORE = 10000
LOSE_SCORE = -20000
SCORE_PER_PIECE = 100
SCORE_PER_CONTROL = 10
CENTER_POSITIONS = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]


class KyawanAgentV3(Agent):
    def __init__(self):
        super().__init__(name="KyawanV3")

    def calc_best_move(
        self,
        board,
        prev_board,
        player,
        time_remaining,
    ):
        if time_remaining < 1000:
            depth = 3
        elif time_remaining < 2000:
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
        # 勝敗の評価
        if is_winner(board, player):
            return WIN_SCORE + depth, None  # 速く勝つほど高評価
        elif is_winner(board, opponent):
            return LOSE_SCORE - depth, None  # 遅く負けるほど高評価

        if depth == 0:
            return self.evaluate(board, player, opponent, depth), None

        best_score = float("-inf")
        best_move = None

        # Move候補に優先順位をつけてソートする
        moves = get_legal_moves(board, player)
        moves.sort(
            key=lambda move: self.calc_move_priority(board, move, player, opponent),
            reverse=True,
        )
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
        score = 0

        # 生存数の評価
        player_count = sum(1 for cell in board if cell == player)
        opponent_count = sum(1 for cell in board if cell == opponent)
        score += (player_count - opponent_count) * SCORE_PER_PIECE

        # 中央支配の評価
        center_positions = CENTER_POSITIONS
        for x, y in center_positions:
            index = xy_to_index(x, y)
            if board[index] == player:
                score += SCORE_PER_CONTROL
            elif board[index] == opponent:
                score -= SCORE_PER_CONTROL

        return score

    def calc_move_priority(
        self,
        board: list[Cell],
        move: Move,
        player: Cell,
        opponent: Cell,
    ) -> int:
        """
        簡単な優先順位付けを行う関数（重い処理は避ける）
        - 相手の駒を押すMoveは優先度高め
        - 中央に近づくMoveは優先度高め
        - その他は優先度低め
        """
        priority = 0

        tx, ty = move.x + move.dx, move.y + move.dy
        if not is_in_board(tx, ty):
            return -100

        # 相手の駒を押すMoveは優先度高め
        target_cell = board[xy_to_index(tx, ty)]
        if target_cell == opponent:
            priority += 100

        # 中央に近づくMoveは優先度高め
        dist_to_center = abs(move.x - 2) + abs(move.y - 2)
        priority += max(0, 4 - dist_to_center)

        return priority
