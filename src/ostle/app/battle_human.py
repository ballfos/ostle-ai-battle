from __future__ import annotations

from typing import Iterable

from ostle.agents.base import Agent
from ostle.agents.inoue_agent.agent import InoueAgent
from ostle.agents.random import RandomAgent
from ostle.core.board import (Cell, Move, applied_move, create_initial_board,
                              get_legal_moves, is_winner)

SYMBOLS = {
	Cell.EMPTY: ".",
	Cell.Player1: "1",
	Cell.Player2: "2",
	Cell.HOLE: "O",
}


def render_board(board: list[Cell]) -> str:
	rows: list[str] = []
	for y in range(5):
		row = []
		for x in range(5):
			row.append(SYMBOLS[board[y * 5 + x]])
		rows.append(" ".join(row))
	return "\n".join(rows)


def format_moves(moves: Iterable[Move]) -> str:
	return ", ".join(f"({m.x},{m.y},{m.dx},{m.dy})" for m in moves)


def parse_move(text: str) -> Move | None:
	try:
		parts = [int(p) for p in text.strip().split()]
	except ValueError:
		return None
	if len(parts) != 4:
		return None
	return Move(parts[0], parts[1], parts[2], parts[3])


def choose_agent(name: str) -> Agent:
	if name.lower() == "random":
		return RandomAgent()
	return InoueAgent()


def main() -> None:
	print("Ostle Human Battle")
	print("入力形式: x y dx dy  (例: 0 0 1 0)")
	print("dx,dy は -1,0,1 のいずれか（斜めは不可）")

	human_side = input("人間は先手(1)か後手(2)か？ [1/2]: ").strip() or "1"
	human_player = Cell.Player1 if human_side == "1" else Cell.Player2

	ai_type = input("AI種類を選択 [inoue/random]: ").strip() or "inoue"
	ai_agent = choose_agent(ai_type)

	board = create_initial_board()
	history: list[tuple[list[Cell], Move]] = []
	turn = Cell.Player1
	time_remaining_ms = 5_000.0

	while True:
		print("\n--- Turn:", turn.name, "---")
		print(render_board(board))

		legal_moves = get_legal_moves(board, turn)
		if not legal_moves:
			winner = turn.opponent()
			print("合法手なし。", winner.name, "の勝ち")
			break

		if turn == human_player:
			print("合法手:", format_moves(legal_moves))
			move = None
			while move is None or move not in legal_moves:
				raw = input("手を入力: ")
				move = parse_move(raw)
				if move is None or move not in legal_moves:
					print("不正な手です。再入力してください。")
		else:
			prev_board = history[-1][0] if history else None
			move = ai_agent.calc_best_move(board, prev_board, turn, time_remaining_ms)
			if move not in legal_moves:
				print("AIが不正な手を返しました。人間の勝ち")
				break
			print("AIの手:", move)

		history.append((board[:], move))
		board = applied_move(board, move)

		if is_winner(board, turn):
			print(render_board(board))
			print(turn.name, "の勝ち")
			break

		turn = turn.opponent()


if __name__ == "__main__":
	main()
