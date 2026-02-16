import sys

import pygame

from ostle.app.engine import AsyncEngine, EngineState
from ostle.core.board import Cell

# --- デザイン定数 (Nord-like Color Palette) ---
# 初期ウィンドウサイズ（起動時）
INIT_WINDOW_WIDTH = 1000
INIT_WINDOW_HEIGHT = 1000

# 盤面のサイズ設定（固定）
GRID_SIZE = 5
CELL_SIZE = 120  # 駒のサイズ
BOARD_PIXEL_SIZE = GRID_SIZE * CELL_SIZE

# 色定義
COLOR_BG = (46, 52, 64)  # 全体の背景 (Dark Blue Grey)
COLOR_BOARD_BG = (59, 66, 82)  # 盤面の背景
COLOR_GRID = (76, 86, 106)  # 格子線
COLOR_HOLE = (30, 30, 35)  # 穴 (ほぼ黒)

# プレイヤーカラー
COLOR_P1_BODY = (94, 129, 172)  # Player1 (Blue)
COLOR_P1_BORDER = (129, 161, 193)
COLOR_P2_BODY = (236, 239, 244)  # Player2 (White/Snow)
COLOR_P2_BORDER = (216, 222, 233)

# テキスト
COLOR_TEXT_MAIN = (236, 239, 244)  # Snow
COLOR_TEXT_SUB = (216, 222, 233)
COLOR_ACCENT = (136, 192, 208)  # Cyan
COLOR_WARN = (191, 97, 106)  # Red


class OstleWindow:
    def __init__(self, engine: AsyncEngine, caption="Ostle AI Battle"):
        self.engine = engine

        # Pygame初期化
        pygame.init()
        # RESIZABLEフラグを立ててウィンドウサイズ変更を許可
        self.screen = pygame.display.set_mode(
            (INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()

        # --- フォント設定 (Mac対応) ---
        font_names = [
            "HYu Gothic",
            "Yu Gothic",
            "Yu Gothic",
            "Meiryo",
            "Arial",
            pygame.font.get_default_font(),
        ]

        self.font_ui = pygame.font.SysFont(font_names, 28)
        self.font_big = pygame.font.SysFont(font_names, 64, bold=True)
        self.font_small = pygame.font.SysFont(font_names, 20)
        self.font_guide = pygame.font.SysFont(font_names, 36, bold=True)

        # 振り返り用のインデックス管理
        self.review_index = -1

        self.show_result_overlay = True

    def run(self):
        """メインループ"""
        running = True

        while running:
            # FPS制御
            dt_ms = self.clock.tick(24)

            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

                # ウィンドウリサイズ時の処理（必須ではないが明示的に書くことも可能）
                if event.type == pygame.VIDEORESIZE:
                    # Pygame2系では自動でsurfaceが更新されるが、
                    # 明示的な再描画などをここに書いても良い
                    pass

                if event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key)

            # エンジンの更新
            self.engine.update(dt_ms)

            # 描画
            self._draw()

        pygame.quit()
        sys.exit()

    def _handle_keydown(self, key):
        """キー入力処理"""
        # スペースキー: ニューゲーム
        if key == pygame.K_SPACE:
            self.engine.reset()
            self.review_index = -1
            self.show_result_overlay = True
            print("--- New Game ---")
            return

        # 矢印キー: 終了後の振り返り
        if self.engine.state == EngineState.FINISHED:
            history_len = len(self.engine.history)
            self.show_result_overlay = False  # 操作したら結果オーバーレイは消す
            if self.review_index == -1:
                self.review_index = history_len

            if key == pygame.K_LEFT:
                self.review_index = max(0, self.review_index - 1)
            elif key == pygame.K_RIGHT:
                self.review_index = min(history_len, self.review_index + 1)

    def _get_display_board(self):
        """描画すべき盤面データの取得"""
        if self.engine.state != EngineState.FINISHED:
            return self.engine.board

        if self.review_index == -1 or self.review_index == len(self.engine.history):
            return self.engine.board

        return self.engine.history[self.review_index][0]

    def _calculate_layout(self):
        """現在のウィンドウサイズに基づいて盤面の左上座標を計算する"""
        w, h = self.screen.get_size()

        # 画面中央に配置
        board_x = (w - BOARD_PIXEL_SIZE) // 2
        board_y = (h - BOARD_PIXEL_SIZE) // 2

        return w, h, board_x, board_y

    def _draw(self):
        """描画メイン"""
        self.screen.fill(COLOR_BG)

        # 動的なレイアウト計算
        screen_w, screen_h, bx, by = self._calculate_layout()

        # 盤面の背景を描画
        pygame.draw.rect(
            self.screen,
            COLOR_BOARD_BG,
            (bx - 10, by - 10, BOARD_PIXEL_SIZE + 20, BOARD_PIXEL_SIZE + 20),
            border_radius=15,
        )

        board = self._get_display_board()

        self._draw_grid(bx, by)
        self._draw_pieces(board, bx, by)
        self._draw_header_ui(screen_w, by)  # 盤面の上(by)を基準にする

        # 終了時は画面下部に操作ガイドを出す
        if self.engine.state == EngineState.FINISHED:
            self._draw_footer_guide(screen_w, screen_h)
            self._draw_result_overlay(screen_w, screen_h)

        pygame.display.flip()

    def _draw_grid(self, start_x, start_y):
        """グリッド線 (動的座標)"""
        for i in range(GRID_SIZE + 1):
            pos = i * CELL_SIZE

            # 縦線
            line_x = start_x + pos
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (line_x, start_y),
                (line_x, start_y + BOARD_PIXEL_SIZE),
                2,
            )

            # 横線
            line_y = start_y + pos
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (start_x, line_y),
                (start_x + BOARD_PIXEL_SIZE, line_y),
                2,
            )

    def _draw_pieces(self, board, start_x, start_y):
        """駒と穴の描画 (動的座標)"""
        for i, cell in enumerate(board):
            # グリッド座標 -> 画面ピクセル座標変換
            grid_x = i % GRID_SIZE
            grid_y = i // GRID_SIZE

            cx = start_x + grid_x * CELL_SIZE + CELL_SIZE // 2
            cy = start_y + grid_y * CELL_SIZE + CELL_SIZE // 2

            # 駒のサイズ
            size = int(CELL_SIZE * 0.85)
            offset = size // 2

            if cell == Cell.HOLE:
                pygame.draw.circle(
                    self.screen, COLOR_HOLE, (cx, cy), int(CELL_SIZE * 0.4)
                )
                pygame.draw.circle(
                    self.screen, (20, 20, 25), (cx, cy), int(CELL_SIZE * 0.4), 4
                )

            elif cell in (Cell.Player1, Cell.Player2):
                is_p1 = cell == Cell.Player1
                base_color = COLOR_P1_BODY if is_p1 else COLOR_P2_BODY
                border_color = COLOR_P1_BORDER if is_p1 else COLOR_P2_BORDER

                # 影
                shadow_rect = (cx - offset + 4, cy - offset + 4, size, size)
                pygame.draw.rect(
                    self.screen, (0, 0, 0, 60), shadow_rect, border_radius=12
                )

                # 本体
                body_rect = (cx - offset, cy - offset, size, size)
                pygame.draw.rect(self.screen, base_color, body_rect, border_radius=12)

                # 枠線
                pygame.draw.rect(
                    self.screen, border_color, body_rect, 3, border_radius=12
                )

                # ラベル
                label = "1" if is_p1 else "2"
                text_color = (255, 255, 255) if is_p1 else (50, 50, 60)
                text = self.font_ui.render(label, True, text_color)
                self.screen.blit(
                    text, (cx - text.get_width() // 2, cy - text.get_height() // 2)
                )

    def _draw_header_ui(self, screen_w, board_top_y):
        """上部のステータスバー (画面幅に応じて配置)"""
        # 盤面の少し上に配置する
        ui_y = max(20, board_top_y - 100)

        # Player 1 (Left) - 真ん中から300px左
        self._draw_player_badge(Cell.Player1, screen_w // 2 - 300, ui_y)

        # Player 2 (Right) - 真ん中から300px右
        self._draw_player_badge(Cell.Player2, screen_w // 2 + 300, ui_y)

        # 中央の状態表示
        center_x = screen_w // 2

        if self.engine.state == EngineState.THINKING:
            status_text = "THINKING..."
            color = COLOR_ACCENT
        elif self.engine.state == EngineState.FINISHED:
            status_text = "FINISH"
            color = COLOR_WARN
        else:
            status_text = "READY"
            color = COLOR_TEXT_SUB

        text = self.font_ui.render(status_text, True, color)
        bg_rect = text.get_rect(center=(center_x, ui_y + 30))
        bg_rect.inflate_ip(40, 20)
        pygame.draw.rect(self.screen, (40, 40, 45), bg_rect, border_radius=8)
        self.screen.blit(text, text.get_rect(center=(center_x, ui_y + 30)))

    def _draw_player_badge(self, player_cell, x, y):
        agent = self.engine.player_agents[player_cell]
        time_ms = self.engine.time_remaining[player_cell]
        is_turn = (self.engine.turn == player_cell) and (
            self.engine.state != EngineState.FINISHED
        )

        width, height = 300, 80
        left = x - width // 2

        rect = (left, y, width, height)
        bg_color = (60, 66, 80) if not is_turn else (70, 80, 100)
        border_color = COLOR_ACCENT if is_turn else (80, 80, 90)

        pygame.draw.rect(self.screen, bg_color, rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=8)

        name_text = self.font_ui.render(agent.name, True, COLOR_TEXT_MAIN)
        self.screen.blit(name_text, (left + 15, y + 5))
        player_text = self.font_small.render(player_cell.name, True, COLOR_TEXT_SUB)
        self.screen.blit(player_text, (left + 15, y + 40))

        time_str = f"{time_ms / 1000:.1f}s"
        time_color = COLOR_ACCENT if time_ms > 10000 else COLOR_WARN
        time_text = self.font_big.render(time_str, True, time_color)
        scaled_time = pygame.transform.smoothscale(
            time_text,
            (int(time_text.get_width() * 0.5), int(time_text.get_height() * 0.5)),
        )
        self.screen.blit(
            scaled_time,
            (
                left + width - scaled_time.get_width() - 10,
                y + height - scaled_time.get_height() - 5,
            ),
        )

    def _draw_footer_guide(self, screen_w, screen_h):
        """画面下部の操作ガイド"""
        current_step = self.review_index
        total_step = len(self.engine.history)
        if current_step == -1:
            current_step = total_step

        # 画面下部から少し上に配置
        guide_y = screen_h - 100

        step_str = f"Turn: {current_step} / {total_step}"
        step_text = self.font_ui.render(step_str, True, COLOR_ACCENT)
        self.screen.blit(
            step_text, (screen_w // 2 - step_text.get_width() // 2, guide_y)
        )

        cmd_str = "[←] Prev   [→] Next   [SPACE] New Game"
        cmd_text = self.font_guide.render(cmd_str, True, COLOR_TEXT_MAIN)

        alpha = 255
        if self.review_index == total_step or self.review_index == -1:
            alpha = (pygame.time.get_ticks() // 5) % 255
        cmd_text.set_alpha(200 if alpha > 100 else 255)

        self.screen.blit(
            cmd_text, (screen_w // 2 - cmd_text.get_width() // 2, guide_y + 40)
        )

    def _draw_result_overlay(self, screen_w, screen_h):
        """結果表示オーバーレイ"""
        if not self.show_result_overlay:
            return
        if self.engine.winner:
            winner_agent = self.engine.player_agents[self.engine.winner]
            title = f"{winner_agent.name} WINS!"
            color = COLOR_ACCENT
        else:
            title = "DRAW"
            color = COLOR_TEXT_SUB

        text = self.font_big.render(title, True, color)

        center_x = screen_w // 2
        center_y = screen_h // 2

        s = pygame.Surface((screen_w, 120))
        s.set_alpha(220)
        s.fill((0, 0, 0))
        self.screen.blit(s, (0, center_y - 60))

        self.screen.blit(
            text, (center_x - text.get_width() // 2, center_y - text.get_height() // 2)
        )
