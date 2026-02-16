from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import pygame

from ostle.agents.base import Agent
from ostle.agents.inoue_agent.agent import InoueAgent
from ostle.agents.inoue_agent.agent2 import InoueAgentKAI
from ostle.agents.inoue_agent.agent3 import InoueDQNAgent
from ostle.agents.kyawan import KyawanAgent
from ostle.agents.kyawan2 import KyawanAgentV2
from ostle.agents.random import RandomAgent

# 利用可能なエージェント一覧
AVAILABLE_AGENTS = {
	"inoue": InoueAgent,
	"inoue2": InoueAgentKAI,
	"inoue3": InoueDQNAgent,
	"random": RandomAgent,
	"kyawan": KyawanAgent,
	"kyawan2": KyawanAgentV2,
}

from ostle.app.engine import AsyncEngine, EngineState
from ostle.battle import battle_window
from ostle.core.board import Cell, Move, get_legal_moves


@dataclass
class HumanInput:
	selected: tuple[int, int] | None = None
	moves: list[Move] | None = None
	message: str | None = None


class HumanAgent(Agent):
	def __init__(self):
		super().__init__(name="Human")
		self._queue: queue.Queue[Move] = queue.Queue()

	def submit_move(self, move: Move) -> None:
		self._queue.put(move)

	def calc_best_move(
		self,
		board,
		prev_board,
		player,
		time_remaining,
	):
		return self._queue.get()


class HumanAsyncEngine(AsyncEngine):
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
		self.ai_thread.daemon = True
		self.ai_thread.start()


class HumanBattleWindow(battle_window.OstleWindow):
	def __init__(
		self,
		engine: AsyncEngine,
		human_player: Cell,
		human_agent: HumanAgent,
		caption: str = "Ostle Human Battle",
	):
		super().__init__(engine, caption=caption)
		self.human_player = human_player
		self.human_agent = human_agent
		self.human_input = HumanInput()

	def run(self):
		running = True
		while running:
			dt_ms = self.clock.tick(24)

			# 人間のターン中は持ち時間をリセット
			if self.engine.turn == self.human_player:
				self.engine.time_remaining[self.human_player] = 1_000000.0

			# エンジン更新（ゲーム進行）
			self.engine.update(dt_ms)

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
					running = False
				if event.type == pygame.KEYDOWN:
					self._handle_keydown(event.key)
				if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
					self._handle_click(event.pos)

			self._draw()

		pygame.quit()
		raise SystemExit

	def _handle_click(self, pos: tuple[int, int]) -> None:
		if self.engine.state == EngineState.FINISHED:
			return
		if self.engine.turn != self.human_player:
			return

		mx, my = pos
		_, _, board_x, board_y = self._calculate_layout()

		if not (board_x <= mx < board_x + battle_window.BOARD_PIXEL_SIZE):
			return
		if not (board_y <= my < board_y + battle_window.BOARD_PIXEL_SIZE):
			return

		x = (mx - board_x) // battle_window.CELL_SIZE
		y = (my - board_y) // battle_window.CELL_SIZE

		board = self.engine.board
		legal_moves = get_legal_moves(board, self.human_player)

		# 選択状態がない場合
		if self.human_input.selected is None:
			# クリックしたマスが駒か穴なら選択
			if any(m.x == x and m.y == y for m in legal_moves):
				self.human_input.selected = (x, y)
				self.human_input.moves = self._moves_for(legal_moves, x, y)
			return

		# 選択状態がある場合
		selected_x, selected_y = self.human_input.selected
		moves = self.human_input.moves or []

		# 矢印をクリックしたか確認
		selected_move = self._pick_move_by_marker(
			moves,
			selected_x,
			selected_y,
			mx,
			my,
			board_x,
			board_y,
		)
		if selected_move is not None and selected_move in legal_moves:
			self.human_agent.submit_move(selected_move)
			self.human_input.selected = None
			self.human_input.moves = None
			self.human_input.message = None
			return

		# 矢印以外をクリック → 同じコマなら解除、別のコマなら切り替え
		if x == selected_x and y == selected_y:
			# 同じコマをクリック → 解除
			self.human_input.selected = None
			self.human_input.moves = None
			self.human_input.message = None
		elif any(m.x == x and m.y == y for m in legal_moves):
			# 別のコマをクリック → 切り替え
			self.human_input.selected = (x, y)
			self.human_input.moves = self._moves_for(legal_moves, x, y)

	def _draw(self):
		# 親クラスの描画をすべて実行（pygame.display.flip() を含む）
		self.screen.fill(battle_window.COLOR_BG)

		# 動的なレイアウト計算
		screen_w, screen_h, bx, by = self._calculate_layout()

		# 盤面の背景を描画
		pygame.draw.rect(
			self.screen,
			battle_window.COLOR_BOARD_BG,
			(bx - 10, by - 10, battle_window.BOARD_PIXEL_SIZE + 20, battle_window.BOARD_PIXEL_SIZE + 20),
			border_radius=15,
		)

		board = self._get_display_board()

		self._draw_grid(bx, by)
		self._draw_pieces(board, bx, by)
		self._draw_header_ui(screen_w, by)

		if self.engine.state == EngineState.FINISHED:
			self._draw_footer_guide(screen_w, screen_h)
			self._draw_result_overlay(screen_w, screen_h)

		# 人間対戦用の描画
		self._draw_selection()
		self._draw_targets()
		self._draw_message()

		pygame.display.flip()

	def _draw_selection(self) -> None:
		if self.human_input.selected is None:
			return
		_, _, board_x, board_y = self._calculate_layout()
		x, y = self.human_input.selected
		rect = pygame.Rect(
			board_x + x * battle_window.CELL_SIZE,
			board_y + y * battle_window.CELL_SIZE,
			battle_window.CELL_SIZE,
			battle_window.CELL_SIZE,
		)
		pygame.draw.rect(self.screen, battle_window.COLOR_ACCENT, rect, 4)

	def _draw_targets(self) -> None:
		if not self.human_input.moves:
			return
		_, _, board_x, board_y = self._calculate_layout()
		if self.human_input.selected is None:
			return
		x, y = self.human_input.selected
		base_cx = board_x + x * battle_window.CELL_SIZE + battle_window.CELL_SIZE // 2
		base_cy = board_y + y * battle_window.CELL_SIZE + battle_window.CELL_SIZE // 2
		
		for move in self.human_input.moves:
			dx, dy = move.dx, move.dy
			arrow_cx = base_cx + dx * int(battle_window.CELL_SIZE * 0.35)
			arrow_cy = base_cy + dy * int(battle_window.CELL_SIZE * 0.35)
			self._draw_arrow(arrow_cx, arrow_cy, dx, dy)

	def _draw_message(self) -> None:
		if not self.human_input.message:
			return
		screen_w, _, _, _ = self._calculate_layout()
		text = self.font_ui.render(self.human_input.message, True, battle_window.COLOR_ACCENT)
		self.screen.blit(text, (screen_w // 2 - text.get_width() // 2, 20))

	@staticmethod
	def _moves_for(moves: list[Move], x: int, y: int) -> list[Move]:
		return [m for m in moves if m.x == x and m.y == y]

	@staticmethod
	def _marker_center(
		x: int,
		y: int,
		dx: int,
		dy: int,
		board_x: int,
		board_y: int,
	) -> tuple[int, int]:
		cell = battle_window.CELL_SIZE
		cx = board_x + x * cell + cell // 2 + dx * (cell // 2)
		cy = board_y + y * cell + cell // 2 + dy * (cell // 2)
		return cx, cy

	def _draw_arrow(self, cx: int, cy: int, dx: int, dy: int) -> None:
		size = int(battle_window.CELL_SIZE * 0.15)
		color = battle_window.COLOR_ACCENT
		
		# 矢印のポイント
		points = []
		if dx == 1:  # 右
			points = [(cx + size, cy), (cx - size // 2, cy - size), (cx - size // 2, cy + size)]
		elif dx == -1:  # 左
			points = [(cx - size, cy), (cx + size // 2, cy - size), (cx + size // 2, cy + size)]
		elif dy == 1:  # 下
			points = [(cx, cy + size), (cx - size, cy - size // 2), (cx + size, cy - size // 2)]
		elif dy == -1:  # 上
			points = [(cx, cy - size), (cx - size, cy + size // 2), (cx + size, cy + size // 2)]
		
		if points:
			pygame.draw.polygon(self.screen, color, points)

	@staticmethod
	def _pick_move_by_marker(
		moves: list[Move],
		x: int,
		y: int,
		mx: int,
		my: int,
		board_x: int,
		board_y: int,
	) -> Move | None:
		base_cx = board_x + x * battle_window.CELL_SIZE + battle_window.CELL_SIZE // 2
		base_cy = board_y + y * battle_window.CELL_SIZE + battle_window.CELL_SIZE // 2
		radius = int(battle_window.CELL_SIZE * 0.25)
		r2 = radius * radius
		
		for move in moves:
			dx, dy = move.dx, move.dy
			arrow_cx = base_cx + dx * int(battle_window.CELL_SIZE * 0.35)
			arrow_cy = base_cy + dy * int(battle_window.CELL_SIZE * 0.35)
			
			dist_x = mx - arrow_cx
			dist_y = my - arrow_cy
			if dist_x * dist_x + dist_y * dist_y <= r2:
				return move
		return None


def choose_agent(name: str) -> Agent:
	agent_class = AVAILABLE_AGENTS.get(name.lower())
	if agent_class is None:
		print(f"利用可能なモデル: {', '.join(AVAILABLE_AGENTS.keys())}")
		raise ValueError(f"不明なモデル: {name}")
	return agent_class()


def main() -> None:
	human_side = input("人間は先手(1)か後手(2)か？ [1/2]: ").strip() or "1"
	human_player = Cell.Player1 if human_side == "1" else Cell.Player2

	print(f"利用可能なモデル: {', '.join(AVAILABLE_AGENTS.keys())}")
	ai_type = input("AI種類を選択 (デフォルト: inoue): ").strip() or "inoue"
	ai_agent = choose_agent(ai_type)

	human_agent = HumanAgent()

	if human_player == Cell.Player1:
		engine = HumanAsyncEngine(human_agent, ai_agent)
	else:
		engine = HumanAsyncEngine(ai_agent, human_agent)

	window = HumanBattleWindow(engine, human_player, human_agent)
	window.run()


if __name__ == "__main__":
	main()
