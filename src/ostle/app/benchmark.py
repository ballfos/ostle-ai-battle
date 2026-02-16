"""
与えられたエージェントを評価するためのベンチマークエンジン
"""

import argparse
import time

from tqdm import tqdm

from ostle.agents.base import Agent
from ostle.agents.inoue_agent.agent import InoueAgent
from ostle.agents.inoue_agent.agent2 import InoueAgentKAI
from ostle.agents.kyawan import KyawanAgent
from ostle.agents.kyawan2 import KyawanAgentV2
from ostle.agents.kyawan3 import KyawanAgentV3
from ostle.agents.random import RandomAgent
from ostle.core.board import Cell, applied_move, create_initial_board, is_winner

AGENTS = {
    "Random": RandomAgent(),
    "Kyawan": KyawanAgent(),
    "KyawanV2": KyawanAgentV2(),
    "KyawanV3": KyawanAgentV3(),
    "Inoue": InoueAgent(),
    "InoueKai": InoueAgentKAI(),
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ostle Agents")
    parser.add_argument(
        "--agent1",
        type=str,
        choices=AGENTS.keys(),
        default="KyawanV3",
        help="First agent to benchmark",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        choices=AGENTS.keys(),
        default="KyawanV2",
        help="Second agent to benchmark",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of games to play for benchmarking",
    )
    args = parser.parse_args()

    agent1 = AGENTS[args.agent1]
    agent2 = AGENTS[args.agent2]

    benchmark(agent1, agent2, num_games=args.num_games)


def benchmark(
    agent1: Agent,
    agent2: Agent,
    num_games: int = 10,
    time_limit_ms: int = 5 * 1000,
) -> dict[Cell, int]:
    results = {
        Cell.Player1: 0,
        Cell.Player2: 0,
        Cell.EMPTY: 0,  # 引き分け
    }

    for i in tqdm(range(num_games), desc="Benchmarking"):
        first_player = Cell.Player1 if i % 2 == 0 else Cell.Player2
        winner = _play_one_game(agent1, agent2, first_player, time_limit_ms)
        results[winner] += 1

    print(f"Player1 {agent1.name} wins: {results[Cell.Player1]}")
    print(f"Player2 {agent2.name} wins: {results[Cell.Player2]}")
    print(f"Draws: {results[Cell.EMPTY]}")


def _play_one_game(
    agent1: Agent,
    agent2: Agent,
    first_player: Cell,
    time_limit_ms: int,
) -> Cell:
    """
    Player1がagent1、Player2がagent2として1ゲームをプレイし、勝者を返す
    引き分けの場合はCell.EMPTYを返す
    """
    player_agents = {
        Cell.Player1: agent1,
        Cell.Player2: agent2,
    }
    turn = first_player
    time_remaining = {
        Cell.Player1: time_limit_ms,
        Cell.Player2: time_limit_ms,
    }

    board = create_initial_board()
    prev_board = None

    while True:
        agent = player_agents[turn]
        start_time = time.time()
        move = agent.calc_best_move(board, prev_board, turn, time_remaining[turn])
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        time_remaining[turn] -= elapsed_time

        if time_remaining[turn] < 0:
            return turn.opponent()

        prev_board = board[:]
        board = applied_move(board, move)
        if board == prev_board:
            return Cell.EMPTY  # No change in board state indicates a draw

        if is_winner(board, turn):
            return turn
        elif is_winner(board, turn.opponent()):
            return turn.opponent()

        turn = turn.opponent()


if __name__ == "__main__":
    main()
