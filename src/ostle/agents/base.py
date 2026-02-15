from abc import ABC, abstractmethod

from ostle.core.board import Cell, Move


class Agent(ABC):
    def __init__(self, name: str):
        self.name = name

    def get_color(self, player: Cell) -> tuple[int, int, int]:
        if player == Cell.Player1:
            return self.color
        elif player == Cell.Player2:
            # 2pは薄める
            r, g, b = self.color
            return (min(255, r + 50), min(255, g + 50), min(255, b + 50))
        else:
            raise ValueError("Invalid player cell")

    @abstractmethod
    def calc_best_move(
        self,
        board: list[Cell],
        prev_board: list[Cell],
        player: Cell,
        time_remaining: float,
    ) -> Move:
        pass
