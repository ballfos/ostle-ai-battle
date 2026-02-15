import queue
import random
import threading
from enum import Enum, auto

from ostle.agents.base import Agent
from ostle.core.board import (
    Cell,
    applied_move,
    create_initial_board,
    is_winner,
)

TIME_LIMIT_MS = 5 * 1000


class EngineState(Enum):
    IDLE = auto()
    THINKING = auto()
    FINISHED = auto()


class AsyncEngine:
    def __init__(
        self,
        agent1: Agent,
        agent2: Agent,
    ):
        self.player_agents = {
            Cell.Player1: agent1,
            Cell.Player2: agent2,
        }

        self.reset()

    def reset(self):
        self.board = create_initial_board()
        self.turn = random.choice([Cell.Player1, Cell.Player2])
        self.time_remaining = {
            Cell.Player1: TIME_LIMIT_MS,
            Cell.Player2: TIME_LIMIT_MS,
        }
        self.history = []
        self.state = EngineState.IDLE
        self.winner = None

        self.move_queue = queue.Queue()
        self.ai_thread = None

    def update(self, dt_ms: float):
        """
        毎フレームごとに呼び出される更新関数

        仕様
        - dt_msは前回のupdateからの経過時間（ミリ秒）
        - Agentが待機中ならば，非同期でcalc_best_moveを呼び出す
        - Agentが思考中ならば，経過時間をtime_remainingに減算する
        - Agentが思考時間を使い切ったら，敗北する
        - Agentの思考が完了したら，QueueからMoveを受け取って盤面を更新する
        """

        match self.state:
            case EngineState.IDLE:
                self.state = EngineState.THINKING
                self._run_ai_thread()
            case EngineState.THINKING:
                self.time_remaining[self.turn] -= dt_ms

                # タイムアウト判定
                if self.time_remaining[self.turn] <= 0:
                    self.on_finish(
                        winner=self.turn.opponent(),
                        reason="timeout",
                    )
                    return

                # Agentの思考が完了しているか判定
                try:
                    move = self.move_queue.get_nowait()
                except queue.Empty:
                    return

                # moveが取得できた場合は盤面に適用してターンを交代する
                try:
                    # move適用前に盤面と履歴を保存
                    self.history.append((self.board[:], move))
                    # moveを適用する
                    self.board = applied_move(self.board, move)

                    # 勝敗判定
                    if is_winner(self.board, self.turn):
                        self.on_finish(
                            winner=self.turn,
                            reason="win_condition",
                        )
                        return

                    # ターン交代
                    self.state = EngineState.IDLE
                    self.turn = self.turn.opponent()
                except Exception:
                    # Agentが合法手を返さなかった場合は敗北
                    self.on_finish(
                        winner=self.turn.opponent(),
                        reason="illegal_move",
                    )

            case EngineState.FINISHED:
                pass

    def _run_ai_thread(self):

        self.ai_thread = threading.Thread(
            target=self._ai_worker,
            args=(
                self.player_agents[self.turn],
                self.board,
                self.history[-1][0] if self.history else None,
                self.turn,
                self.time_remaining[self.turn],
            ),
        )
        self.ai_thread.start()

    def _ai_worker(
        self,
        agent: Agent,
        board: list[Cell],
        prev_board: list[Cell] | None,
        player: Cell,
        time_remaining: float,
    ):
        """
        Agentのcalc_best_moveを非同期で実行するワーカー関数
        結果はmove_queueに格納される
        """

        best_move = agent.calc_best_move(board, prev_board, player, time_remaining)
        self.move_queue.put(best_move)

    def on_finish(self, winner: Cell, reason: str):
        """
        ゲーム終了時のコールバック関数
        - winner: 勝者のCell（Cell.BLACK, Cell.WHITE, またはNone）
        - reason: 終了理由の文字列（例: "timeout", "illegal_move", "win_condition"など）
        """

        self.state = EngineState.FINISHED
        self.winner = winner
        self.reason = reason
        print(f"Game finished! Winner: {winner}, Reason: {reason}")
