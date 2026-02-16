from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

# 既存のモジュール構造を維持
from ostle.agents.inoue_agent.agent3 import ACTION_SIZE, DIRS, DQNModel
from ostle.core.board import (BOARD_WIDTH, Cell, Move, applied_move,
                              create_initial_board, get_legal_moves, is_winner)

INPUT_SIZE = 3 * BOARD_WIDTH * BOARD_WIDTH


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 50_000
    max_steps_per_episode: int = 200
    batch_size: int = 512
    gamma: float = 0.99
    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    start_training_after: int = 10_000
    train_interval: int = 1
    target_update_interval: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_000_000
    
    hidden_sizes: tuple[int, ...] = (1024, 512, 512)
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "src/ostle/agents/inoue_agent/artifacts"
    seed: int = 42


@dataclass(frozen=True)
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    next_legal_actions: list[int]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def encode_board(board: list[Cell], player: Cell, device: torch.device) -> torch.Tensor:
    opponent = player.opponent()
    planes = torch.zeros((3, BOARD_WIDTH, BOARD_WIDTH), dtype=torch.float32)
    for y in range(BOARD_WIDTH):
        for x in range(BOARD_WIDTH):
            cell = board[y * BOARD_WIDTH + x]
            if cell == player:
                planes[0, y, x] = 1.0  # 自分
            elif cell == opponent:
                planes[1, y, x] = 1.0  # 相手
            elif cell == Cell.HOLE:
                planes[2, y, x] = 1.0  # 穴
    return planes.flatten().unsqueeze(0).to(device)


def move_to_action(move: Move) -> int:
    dir_index = DIRS.index((move.dx, move.dy))
    return (move.y * BOARD_WIDTH + move.x) * len(DIRS) + dir_index


def legal_actions(moves: Iterable[Move]) -> list[int]:
    return [move_to_action(move) for move in moves]


def select_action(
    policy_net: nn.Module,
    state: torch.Tensor,
    moves: list[Move],
    epsilon: float,
) -> tuple[int, Move]:
    if random.random() < epsilon:
        move = random.choice(moves)
        return move_to_action(move), move

    with torch.no_grad():
        q_values = policy_net(state).squeeze(0)
    
    best_move = None
    best_score = float("-inf")
    best_action = None
    
    # 合法手の中から最大のQ値を持つアクションを選択
    for move in moves:
        action = move_to_action(move)
        score = q_values[action].item()
        if score > best_score:
            best_score = score
            best_move = move
            best_action = action
            
    if best_move is None:
        move = random.choice(moves)
        return move_to_action(move), move
        
    return best_action, best_move


def compute_epsilon(config: TrainConfig, step: int) -> float:
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    ratio = step / config.epsilon_decay_steps
    return config.epsilon_start + ratio * (config.epsilon_end - config.epsilon_start)


def optimize_model(
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    config: TrainConfig,
    device: torch.device,
) -> float:
    if len(buffer) < config.batch_size:
        return 0.0

    batch = buffer.sample(config.batch_size)
    states = torch.cat([t.state for t in batch]).to(device)
    actions = torch.tensor([t.action for t in batch], device=device)
    rewards = torch.tensor([t.reward for t in batch], device=device)
    next_states = torch.cat([t.next_state for t in batch]).to(device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

    # Q(s, a)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # 次の状態 next_state は「相手の手番」です。
        # ゼロサムゲームにおいて、相手は自分(現在の学習主体)にとって最悪の手(相手にとって最善の手)を選びます。
        # したがって、ターゲットは r + gamma * (- max Q_opponent(s', a')) となります。
        
        next_q_all = target_net(next_states)
        max_next_q = []
        for idx, transition in enumerate(batch):
            if transition.done or not transition.next_legal_actions:
                max_next_q.append(0.0)
                continue
            
            # 合法手のみから最大値を取得
            legal_q = next_q_all[idx, transition.next_legal_actions]
            max_next_q.append(legal_q.max().item())
            
        max_next_q_tensor = torch.tensor(max_next_q, device=device)
        
        # 【重要修正】対戦ゲーム用に符号を反転させる (Minimaxの考え方)
        # 相手が勝つ状態(=相手のQが高い)は、自分にとって価値が低い(=マイナス)
        targets = rewards + (1.0 - dones) * config.gamma * (-max_next_q_tensor)

    # Huber Loss の方が外れ値（極端なQ値）に対して安定する場合があるが、MSEでも可
    loss = nn.MSELoss()(q_values, targets)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 勾配クリッピング（学習安定化のおまじない）
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    
    optimizer.step()
    return loss.item()


def train(config: TrainConfig | None = None) -> None:
    config = config or TrainConfig()
    
    # 再現性の確保
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    print(f"Training on device: {device}")

    policy_net = DQNModel(INPUT_SIZE, list(config.hidden_sizes), ACTION_SIZE).to(device)
    target_net = DQNModel(INPUT_SIZE, list(config.hidden_sizes), ACTION_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_size)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    
    for episode in range(1, config.episodes + 1):
        board = create_initial_board()
        # 先攻後攻をランダムにして、偏りを防ぐ
        player = Cell.Player1 if random.random() < 0.5 else Cell.Player2
        episode_reward = 0.0

        for _ in range(config.max_steps_per_episode):
            legal_moves = get_legal_moves(board, player)
            
            # 動ける手がない＝負け
            if not legal_moves:
                reward = -1.0 # 敗北報酬
                state = encode_board(board, player, device)
                
                # 次の状態はないが、形式的に埋める
                dummy_next = torch.zeros_like(state)
                buffer.push(
                    Transition(
                        state=state.cpu(),
                        action=0, # ダミー
                        reward=reward,
                        next_state=dummy_next.cpu(),
                        done=True,
                        next_legal_actions=[],
                    )
                )
                episode_reward += reward
                break

            epsilon = compute_epsilon(config, global_step)
            state = encode_board(board, player, device)
            action, move = select_action(policy_net, state, legal_moves, epsilon)

            next_board = applied_move(board, move)
            
            reward = 0.0
            done = False
            
            # 勝敗判定
            if is_winner(next_board, player):
                reward = 1.0 # 勝利報酬
                done = True
            elif is_winner(next_board, player.opponent()):
                # 自分の手で相手が勝つ状態になった（自殺手など）
                reward = -1.0 
                done = True

            next_player = player.opponent()
            
            # 相手視点の盤面エンコード
            next_state = encode_board(next_board, next_player, device)
            next_moves = get_legal_moves(next_board, next_player)
            
            buffer.push(
                Transition(
                    state=state.cpu(),
                    action=action,
                    reward=reward,
                    next_state=next_state.cpu(),
                    done=done,
                    next_legal_actions=legal_actions(next_moves),
                )
            )

            episode_reward += reward
            board = next_board
            player = next_player
            global_step += 1

            if (
                len(buffer) >= config.start_training_after
                and global_step % config.train_interval == 0
            ):
                optimize_model(
                    policy_net,
                    target_net,
                    optimizer,
                    buffer,
                    config,
                    device,
                )

            if global_step % config.target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        
        # 進捗表示
        if episode % 1000 == 0:
            print(f"Episode {episode}/{config.episodes}, Step: {global_step}, Epsilon: {epsilon:.3f}")

    # 最終モデルの保存
    torch.save(policy_net.state_dict(), save_dir / "dqn_strongest.pt")
    print("Training finished.")

def main() -> None:
    train()

if __name__ == "__main__":
    main()