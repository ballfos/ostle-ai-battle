import time
from ostle.agents.base import Agent
from ostle.core.board import (
    Cell, Move, applied_move, get_legal_moves, 
    is_same_board, is_winner
)

class TheVoidAgent(Agent):
    def __init__(self):
        super().__init__(name="TheVoid")

    def calc_best_move(self, board, prev_board, player, time_remaining_ms):
        # 1. 1手で勝てるなら即座に打つ（探索の節約）
        legal_moves = get_legal_moves(board, player)
        if not legal_moves: return None
        
        for m in legal_moves:
            if is_winner(applied_move(board, m), player):
                return m

        # 2. 残り時間に応じて深さを固定
        # Pythonの場合、安全圏は深さ3。100ms以下なら深さ2。
        depth = 3 if time_remaining_ms > 500 else 2
        
        # 3. 探索開始
        _, best_move = self._minimax(board, prev_board, player, depth, -99999, 99999)
        return best_move if best_move else legal_moves[0]

    def _minimax(self, board, prev_board, player, depth, alpha, beta):
        opp = player.opponent()
        
        # 終端評価
        if is_winner(board, player): return 10000 + depth, None
        if is_winner(board, opp): return -10000 - depth, None
        if depth == 0:
            return self.evaluate(board, player), None

        moves = get_legal_moves(board, player)
        best_m = None
        
        # 4. 簡易的な並べ替え（相手の駒を動かす手を優先）
        # これにより枝刈り（Alpha-Beta）が劇的に効く
        moves.sort(key=lambda m: 0 if board[m.y*5+m.x + m.dx + m.dy*5 if 0<=m.x+m.dx<5 and 0<=m.y+m.dy<5 else 0] == opp else 1)

        for m in moves:
            next_b = applied_move(board, m)
            if prev_board and is_same_board(next_b, prev_board):
                continue
            
            val, _ = self._minimax(next_b, board, opp, depth - 1, -beta, -alpha)
            val = -val

            if val > alpha:
                alpha = val
                best_m = m
            if alpha >= beta:
                break
        
        return alpha, best_m

    def evaluate(self, board, player):
        # 高速評価関数：1回のループで全てを計算
        opp = player.opponent()
        score = 0
        p_count = 0
        o_count = 0
        
        for i in range(25):
            c = board[i]
            if c == player:
                p_count += 1
                # 中央付近なら加点 (x=1~3, y=1~3)
                x, y = i % 5, i // 5
                if 0 < x < 4 and 0 < y < 4: score += 10
            elif c == opp:
                o_count += 1
                x, y = i % 5, i // 5
                if 0 < x < 4 and 0 < y < 4: score -= 10
        
        # 駒の数に圧倒的な重みをつける
        score += (p_count - o_count) * 1000
        return score