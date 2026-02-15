import pytest

from ostle.core.board import (
    BOARD_WIDTH,
    HOLE_POSITION,
    Cell,
    Move,
    applied_move,
    create_initial_board,
    get_legal_moves,
    xy_to_index,
)

# --- フィクスチャ（テスト用データの準備） ---


@pytest.fixture
def empty_board():
    """全て空のボードを返す"""
    return [Cell.EMPTY] * (BOARD_WIDTH * BOARD_WIDTH)


def set_cells(board, cells_dict):
    """
    指定した座標にセルを配置するヘルパー
    cells_dict = {(x, y): Cell.BLACK, ...}
    """
    for (x, y), cell in cells_dict.items():
        board[xy_to_index(x, y)] = cell
    return board


# --- テストケース ---


def test_initial_board():
    """初期盤面の配置テスト"""
    board = create_initial_board()
    assert len(board) == 25

    # 黒の配置チェック (最上段)
    for x in range(BOARD_WIDTH):
        assert board[xy_to_index(x, 0)] == Cell.Player1

    # 白の配置チェック (最下段)
    for x in range(BOARD_WIDTH):
        assert board[xy_to_index(x, 4)] == Cell.Player2

    # 穴の配置チェック (中央)
    hx, hy = HOLE_POSITION
    assert board[xy_to_index(hx, hy)] == Cell.HOLE

    # それ以外はEmpty
    assert board[xy_to_index(0, 1)] == Cell.EMPTY


def test_legal_moves_basic(empty_board):
    """基本的な合法手の生成テスト"""
    # (2, 2)に黒駒を置く
    board = set_cells(empty_board, {(2, 2): Cell.Player1})

    moves = get_legal_moves(board, Cell.Player1)

    # 上下左右4方向に行けるはず
    expected_moves = {
        (2, 2, 0, -1),  # 上
        (2, 2, 0, 1),  # 下
        (2, 2, -1, 0),  # 左
        (2, 2, 1, 0),  # 右
    }

    # Move型をタプルに変換して比較
    actual_moves = set((m.x, m.y, m.dx, m.dy) for m in moves)
    assert actual_moves == expected_moves


def test_legal_moves_hole(empty_board):
    """穴の移動テスト"""
    # (2, 2)に穴、(2, 1)に黒駒、他は空
    board = set_cells(empty_board, {(2, 2): Cell.HOLE, (2, 1): Cell.Player1})

    # 穴の移動候補を取得（プレイヤーは何でも良いがBLACKとする）
    # 穴は自分自身の移動が可能
    moves = get_legal_moves(board, Cell.Player1)
    hole_moves = [m for m in moves if board[xy_to_index(m.x, m.y)] == Cell.HOLE]

    # 上(2, 1)には黒駒があるので行けない。左右下は空なので行けるはず。
    actual_deltas = set((m.dx, m.dy) for m in hole_moves)
    expected_deltas = {(-1, 0), (1, 0), (0, 1)}  # 左、右、下

    assert actual_deltas == expected_deltas


def test_applied_move_simple(empty_board):
    """単純な移動（スライド）"""
    board = set_cells(empty_board, {(1, 1): Cell.Player1})
    move = Move(1, 1, 1, 0)  # 右へ

    new_board = applied_move(board, move)

    assert new_board[xy_to_index(1, 1)] == Cell.EMPTY  # 元の場所
    assert new_board[xy_to_index(2, 1)] == Cell.Player1  # 移動先


def test_applied_move_push_single(empty_board):
    """1つ押す"""
    # [黒][白][空] -> 右へ -> [空][黒][白]
    board = set_cells(empty_board, {(0, 0): Cell.Player1, (1, 0): Cell.Player2})
    move = Move(0, 0, 1, 0)  # (0,0)の黒が右へ動く

    new_board = applied_move(board, move)

    assert new_board[xy_to_index(0, 0)] == Cell.EMPTY
    assert new_board[xy_to_index(1, 0)] == Cell.Player1
    assert new_board[xy_to_index(2, 0)] == Cell.Player2


def test_applied_move_push_chain(empty_board):
    """連鎖押し（3つ並び）"""
    # [黒][白][黒][空] -> 右へ -> [空][黒][白][黒]
    board = set_cells(
        empty_board, {(0, 0): Cell.Player1, (1, 0): Cell.Player2, (2, 0): Cell.Player1}
    )
    move = Move(0, 0, 1, 0)

    new_board = applied_move(board, move)

    assert new_board[xy_to_index(0, 0)] == Cell.EMPTY
    assert new_board[xy_to_index(1, 0)] == Cell.Player1
    assert new_board[xy_to_index(2, 0)] == Cell.Player2
    assert new_board[xy_to_index(3, 0)] == Cell.Player1


def test_applied_move_push_off_board(empty_board):
    """盤外への押し出し（消滅）"""
    # 盤面の右端でテスト
    # [空][黒][白] (x=3, 4) -> 右へ -> [空][空][黒] (白は消滅)
    board = set_cells(empty_board, {(3, 0): Cell.Player1, (4, 0): Cell.Player2})
    move = Move(3, 0, 1, 0)  # 右へ

    new_board = applied_move(board, move)

    assert new_board[xy_to_index(3, 0)] == Cell.EMPTY
    assert new_board[xy_to_index(4, 0)] == Cell.Player1
    # 白は消えていること（盤外判定のロジックによるが、リスト外には出ないので数は減る）
    assert new_board.count(Cell.Player2) == 0


def test_applied_move_into_hole(empty_board):
    """穴への落下（消滅）"""
    # [黒][白][穴] -> 右へ -> [空][黒][穴] (白は消滅)
    # 穴は(2, 2)にあると仮定（フィクスチャで設定）
    board = set_cells(
        empty_board, {(0, 2): Cell.Player1, (1, 2): Cell.Player2, (2, 2): Cell.HOLE}
    )
    move = Move(0, 2, 1, 0)  # 右へ

    new_board = applied_move(board, move)

    assert new_board[xy_to_index(0, 2)] == Cell.EMPTY
    assert new_board[xy_to_index(1, 2)] == Cell.Player1
    assert new_board[xy_to_index(2, 2)] == Cell.HOLE  # 穴はそのまま
    assert new_board.count(Cell.Player2) == 0  # 白は消滅


def test_applied_move_hole_swap(empty_board):
    """穴自体の移動"""
    # [空][穴] -> 左へ -> [穴][空]
    board = set_cells(empty_board, {(1, 0): Cell.HOLE})
    move = Move(1, 0, -1, 0)  # (1,0)の穴を左へ

    new_board = applied_move(board, move)

    assert new_board[xy_to_index(1, 0)] == Cell.EMPTY
    assert new_board[xy_to_index(0, 0)] == Cell.HOLE


def test_immutability(empty_board):
    """元のボードが変更されていないことの確認"""
    board = set_cells(empty_board, {(0, 0): Cell.Player1})
    original_board_state = board[:]

    move = Move(0, 0, 1, 0)
    _ = applied_move(board, move)

    # applied_move呼び出し後も元のboardリストの中身が変わっていないこと
    assert board == original_board_state
