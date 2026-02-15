"""
ボードゲームOstleのゲームロジック

処理の高速化ため
- Boardクラスは実装せず，関数ベースで実装する
- ボードの状態は1次元リストで表現する
- ボードは積極的にコピーして不変オブジェクトとして扱う
"""

from enum import IntEnum, auto
from typing import NamedTuple

BOARD_WIDTH = 5
BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH
HOLE_POSITION = (2, 2)
BLACK_ROW = 0
WHITE_ROW = BOARD_WIDTH - 1


class Cell(IntEnum):
    EMPTY = auto()
    Player1 = auto()
    Player2 = auto()
    HOLE = auto()

    def opponent(self):
        if self == Cell.Player1:
            return Cell.Player2
        elif self == Cell.Player2:
            return Cell.Player1
        else:
            raise ValueError("Cell.EMPTY and Cell.HOLE do not have opponents")


class Move(NamedTuple):
    x: int
    y: int
    dx: int
    dy: int


def create_initial_board() -> list[int]:
    board = [Cell.EMPTY] * BOARD_SIZE

    # 駒の配置
    for x in range(BOARD_WIDTH):
        board[xy_to_index(x, BLACK_ROW)] = Cell.Player1
        board[xy_to_index(x, WHITE_ROW)] = Cell.Player2

    # 穴の配置
    hx, hy = HOLE_POSITION
    board[xy_to_index(hx, hy)] = Cell.HOLE

    return board


def copy_board(board: list[Cell]) -> list[Cell]:
    return board[:]


def get_legal_moves(board: list[Cell], player: Cell) -> list[Move]:
    """
    仕様
    - playerはBLACKまたはWHITEであること
    - playerの駒またはHOLEは移動可能
    - 千日手の判定は行わない
    """

    moves = []

    for y in range(BOARD_WIDTH):
        for x in range(BOARD_WIDTH):
            cell = board[xy_to_index(x, y)]
            if cell == player:
                # 駒の移動
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if not is_in_board(nx, ny):
                        continue
                    moves.append(Move(x, y, dx, dy))

            elif cell == Cell.HOLE:
                # HOLEの移動
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    move = Move(x, y, dx, dy)
                    nx, ny = x + dx, y + dy
                    if not is_in_board(nx, ny):
                        continue
                    next_cell = board[xy_to_index(nx, ny)]
                    if next_cell is not Cell.EMPTY:
                        continue
                    moves.append(move)

    return moves


def applied_move(board: list[Cell], move: Move) -> list[Cell]:
    """
    moveを適用した新しいボードを返す
    moveは合法手であることを前提とする

    仕様
    - 安全のためのエラーチェックは行わない
    - move対象がHOLEならば，move先がEMPTYである前提で入れ替える
    - move対象が駒ならば，EMPTY/HOLE/盤外のいずれかが現れるまで連鎖的に移動する
    """

    new_board = board[:]

    # move対象のセルを取得
    target_cell = new_board[xy_to_index(move.x, move.y)]

    if target_cell == Cell.HOLE:
        # HOLEの移動
        new_x = move.x + move.dx
        new_y = move.y + move.dy
        new_board[xy_to_index(move.x, move.y)] = Cell.EMPTY
        new_board[xy_to_index(new_x, new_y)] = Cell.HOLE

    elif target_cell in (Cell.Player1, Cell.Player2):
        # 駒の移動
        cur_x, cur_y = move.x, move.y
        prev_cell = Cell.EMPTY
        while True:
            new_x = cur_x + move.dx
            new_y = cur_y + move.dy
            if (
                not is_in_board(new_x, new_y)
                or new_board[xy_to_index(new_x, new_y)] == Cell.HOLE
            ):
                new_board[xy_to_index(cur_x, cur_y)] = prev_cell
                break

            if new_board[xy_to_index(new_x, new_y)] == Cell.EMPTY:
                new_board[xy_to_index(new_x, new_y)] = new_board[
                    xy_to_index(cur_x, cur_y)
                ]
                new_board[xy_to_index(cur_x, cur_y)] = prev_cell
                break

            if new_board[xy_to_index(new_x, new_y)] in (Cell.Player1, Cell.Player2):
                tmp_cell = new_board[xy_to_index(cur_x, cur_y)]
                new_board[xy_to_index(cur_x, cur_y)] = prev_cell
                prev_cell = tmp_cell
                cur_x, cur_y = new_x, new_y

    return new_board


def is_winner(board: list[Cell], player: Cell) -> bool:
    """
    playerが勝利しているかを判定する
    仕様
    - playerはBLACKまたはWHITEであること
    - 相手の駒が2つ消えていたら勝利（つまり相手の駒が3つ以下なら勝利）
    """

    opponent = player.opponent()
    opponent_count = sum(1 for cell in board if cell == opponent)
    return opponent_count <= 3


def xy_to_index(x: int, y: int) -> int:
    return y * BOARD_WIDTH + x


def is_in_board(x: int, y: int) -> bool:
    return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_WIDTH
