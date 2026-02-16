from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ostle.agents.base import Agent
from ostle.core.board import BOARD_WIDTH, Cell, Move, get_legal_moves

DIRS: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_SIZE = BOARD_WIDTH * BOARD_WIDTH * len(DIRS)


class DQNModel(nn.Module):
	def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
		super().__init__()
		layers: list[nn.Module] = []
		prev_size = input_size
		for hidden in hidden_sizes:
			layers.append(nn.Linear(prev_size, hidden))
			layers.append(nn.ReLU())
			prev_size = hidden
		layers.append(nn.Linear(prev_size, output_size))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


@dataclass(frozen=True)
class DQNAgentConfig:
	model_path: str | None = "src/ostle/agents/inoue_agent/artifacts/dqn_strongest.pt"
	epsilon: float = 0.0
	hidden_sizes: tuple[int, int] = (128, 128)
	device: str = "cpu"


class InoueDQNAgent(Agent):
	def __init__(self, config: DQNAgentConfig | None = None):
		super().__init__(name="InoueDQNAgent")
		self.config = config or DQNAgentConfig()
		self.device = torch.device(self.config.device)

		input_size = 3 * BOARD_WIDTH * BOARD_WIDTH
		self.model = DQNModel(
			input_size=input_size,
			hidden_sizes=list(self.config.hidden_sizes),
			output_size=ACTION_SIZE,
		).to(self.device)
		self.model.eval()

		if self.config.model_path:
			self._load_weights(self.config.model_path)

	def calc_best_move(
		self,
		board,
		prev_board,
		player,
		time_remaining,
	) -> Move:
		legal_moves = get_legal_moves(board, player)
		if not legal_moves:
			raise ValueError("No legal moves available")

		if random.random() < self.config.epsilon:
			return random.choice(legal_moves)

		q_values = self._predict_q_values(board, player)
		best_move = self._select_best_legal_move(q_values, legal_moves)
		if best_move is None:
			return random.choice(legal_moves)
		return best_move

	def _predict_q_values(self, board: list[Cell], player: Cell) -> torch.Tensor:
		with torch.no_grad():
			state = self._encode_board(board, player)
			q_values = self.model(state)
		return q_values.squeeze(0)

	def _encode_board(self, board: list[Cell], player: Cell) -> torch.Tensor:
		opponent = player.opponent()
		planes = torch.zeros(
			(3, BOARD_WIDTH, BOARD_WIDTH), dtype=torch.float32, device=self.device
		)
		for y in range(BOARD_WIDTH):
			for x in range(BOARD_WIDTH):
				cell = board[y * BOARD_WIDTH + x]
				if cell == player:
					planes[0, y, x] = 1.0
				elif cell == opponent:
					planes[1, y, x] = 1.0
				elif cell == Cell.HOLE:
					planes[2, y, x] = 1.0
		return planes.flatten().unsqueeze(0)

	def _select_best_legal_move(
		self, q_values: torch.Tensor, legal_moves: list[Move]
	) -> Move | None:
		best_score = float("-inf")
		best_move = None
		for move in legal_moves:
			action = self._move_to_action(move)
			score = q_values[action].item()
			if score > best_score:
				best_score = score
				best_move = move
		return best_move

	def _move_to_action(self, move: Move) -> int:
		dir_index = DIRS.index((move.dx, move.dy))
		return (move.y * BOARD_WIDTH + move.x) * len(DIRS) + dir_index

	def _load_weights(self, model_path: str) -> None:
		path = Path(model_path)
		if not path.exists():
			raise FileNotFoundError(f"Model file not found: {model_path}")
		state = torch.load(path, map_location=self.device)
		try:
			self.model.load_state_dict(state)
			self.model.eval()
			return
		except RuntimeError:
			pass

		layer_indices: list[int] = []
		for key in state.keys():
			if not key.startswith("net.") or not key.endswith(".weight"):
				continue
			parts = key.split(".")
			if len(parts) < 3:
				continue
			try:
				layer_indices.append(int(parts[1]))
			except ValueError:
				continue
		layer_indices = sorted(set(layer_indices))

		if not layer_indices:
			raise RuntimeError("Failed to infer model architecture from state_dict")

		layer_shapes = []
		for idx in layer_indices:
			weight = state.get(f"net.{idx}.weight")
			if weight is None:
				continue
			layer_shapes.append(weight.shape)

		if not layer_shapes:
			raise RuntimeError("No linear layers found in state_dict")

		input_size = layer_shapes[0][1]
		output_size = layer_shapes[-1][0]
		hidden_sizes = [shape[0] for shape in layer_shapes[:-1]]

		self.model = DQNModel(
			input_size=input_size,
			hidden_sizes=hidden_sizes,
			output_size=output_size,
		).to(self.device)
		self.model.load_state_dict(state)
		self.model.eval()
