"""
DQN with experience replay for Atari Pong.

Follows "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013,
arXiv:1312.5602) for the network, preprocessing, frame skipping, reward
clipping, epsilon-greedy exploration and the two evaluation metrics
(average reward per episode and average max Q on a fixed set of held-out
states).

Two modes:
  default       Practical settings that make Pong learn in a reasonable time
                (target network, Adam, Huber loss, update every 4 steps).
  --paper-mode  Settings closest to the 2013 paper (no target network,
                RMSProp, squared error, update every step, epsilon annealed
                over 1M steps to 0.1).

Usage:
  python dqn_pong.py                      # practical defaults, 2M steps
  python dqn_pong.py --paper-mode         # 2013 paper settings
  python dqn_pong.py --total-steps 20000  # quick smoke test
"""

import argparse
import csv
import os
import random
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def make_env(env_id: str, seed: int, render_mode=None) -> gym.Env:
    """Create an Atari env with the preprocessing described in Section 4.1.

    frameskip=1 and repeat_action_probability=0.0 on the base env, so that
    frame skipping is done once (by AtariPreprocessing) and the emulator is
    deterministic, as in the 2013 setup.
    """
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode,
    )
    # Grayscale, resize to 84x84, repeat each action for 4 frames.
    # Note: the paper downsamples to 110x84 and crops an 84x84 region;
    # this wrapper resizes the full frame to 84x84 instead.
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )
    # Stack the last 4 frames -> observation shape (4, 84, 84), uint8.
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Network (Section 4.1 of the paper)
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Input 4x84x84 -> Conv(16, 8x8, s4) -> Conv(32, 4x4, s2) -> FC(256) -> Q per action."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(32 * 9 * 9, 256)
        self.out = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0  # uint8 pixels -> [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.flatten(start_dim=1)))
        return self.out(x)


# ---------------------------------------------------------------------------
# Experience replay (Algorithm 1)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-size circular buffer of transitions (s, a, r, s', done).

    To save memory, only the newest frame of s' is stored. Because the
    frames are stacked, s' equals s shifted by one frame plus that new frame.
    One transition takes about 35 KB, so 100k transitions use about 3.5 GB.
    """

    def __init__(self, capacity: int, obs_shape=(4, 84, 84)):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_frame = np.zeros((capacity, *obs_shape[1:]), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.next_frame[self.pos] = next_obs[-1]
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = self.obs[idx]
        next_obs = np.concatenate([obs[:, 1:], self.next_frame[idx][:, None]], axis=1)
        return (
            torch.from_numpy(obs).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(next_obs).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(step / decay_steps, 1.0)
    return start + fraction * (end - start)


def select_action(q_net, obs, epsilon, num_actions, device) -> int:
    if random.random() < epsilon:
        return random.randrange(num_actions)
    with torch.no_grad():
        obs_t = torch.from_numpy(np.asarray(obs)).unsqueeze(0).to(device)
        return int(q_net(obs_t).argmax(dim=1).item())


def collect_holdout_states(env_id: str, seed: int, n: int) -> np.ndarray:
    """Random-policy states used to track average max Q (Section 5.1)."""
    env = make_env(env_id, seed + 1000)
    obs, _ = env.reset()
    states = []
    while len(states) < n:
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        states.append(np.asarray(obs))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.stack(states)


@torch.no_grad()
def average_max_q(q_net, states: np.ndarray, device) -> float:
    values = []
    for i in range(0, len(states), 256):
        batch = torch.from_numpy(states[i:i + 256]).to(device)
        values.append(q_net(batch).max(dim=1).values)
    return float(torch.cat(values).mean().item())


def evaluate(q_net, env_id, seed, episodes, epsilon, device) -> float:
    """Average total reward per episode with a small epsilon (paper uses 0.05)."""
    env = make_env(env_id, seed + 2000)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action = select_action(q_net, obs, epsilon, env.action_space.n, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DQN with experience replay on Atari")
    p.add_argument("--env-id", default="ALE/Pong-v5")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=2_000_000,
                   help="agent steps (each step = 4 emulator frames)")
    p.add_argument("--buffer-size", type=int, default=100_000,
                   help="paper uses 1,000,000; ~35 KB RAM per transition")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update", type=int, default=1_000,
                   help="steps between target network syncs; 0 disables the target network")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-steps", type=int, default=250_000)
    p.add_argument("--optimizer", choices=["adam", "rmsprop"], default="adam")
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--eval-epsilon", type=float, default=0.05)
    p.add_argument("--holdout-states", type=int, default=500)
    p.add_argument("--out-dir", default="runs/pong")
    p.add_argument("--device", default="auto")
    p.add_argument("--paper-mode", action="store_true",
                   help="use settings closest to the 2013 paper")
    args = p.parse_args()

    if args.paper_mode:
        args.target_update = 0       # no target network in the 2013 paper
        args.optimizer = "rmsprop"
        args.loss = "mse"
        args.train_freq = 1          # one minibatch update per agent step
        args.eps_end = 0.1
        args.eps_decay_steps = 1_000_000
        args.lr = 2.5e-4             # the 2013 paper does not state its learning rate
    return args


# ---------------------------------------------------------------------------
# Training loop (Algorithm 1: deep Q-learning with experience replay)
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env(args.env_id, args.seed)
    num_actions = env.action_space.n
    print(f"env={args.env_id} actions={num_actions} device={device} paper_mode={args.paper_mode}")

    q_net = QNetwork(num_actions).to(device)
    use_target = args.target_update > 0
    target_net = QNetwork(num_actions).to(device) if use_target else q_net
    if use_target:
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.RMSprop(q_net.parameters(), lr=args.lr, alpha=0.95, eps=0.01)

    buffer = ReplayBuffer(args.buffer_size)
    holdout = collect_holdout_states(args.env_id, args.seed, args.holdout_states)

    train_log = open(os.path.join(args.out_dir, "train_episodes.csv"), "w", newline="")
    eval_log = open(os.path.join(args.out_dir, "eval.csv"), "w", newline="")
    train_writer = csv.writer(train_log)
    eval_writer = csv.writer(eval_log)
    train_writer.writerow(["step", "episode", "return", "length", "epsilon"])
    eval_writer.writerow(["step", "eval_return", "avg_max_q"])

    obs, _ = env.reset()
    episode, ep_return, ep_len = 0, 0.0, 0
    recent_returns = []
    best_eval = -float("inf")
    start = time.time()

    for step in range(1, args.total_steps + 1):
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)
        action = select_action(q_net, obs, epsilon, num_actions, device)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Reward clipping (Section 5): positive -> +1, negative -> -1, zero unchanged.
        clipped = float(np.sign(reward))
        # Only true termination cuts the bootstrap; a time-limit truncation does not.
        buffer.add(np.asarray(obs), action, clipped, np.asarray(next_obs), float(terminated))

        obs = next_obs
        ep_return += reward
        ep_len += 1

        if terminated or truncated:
            episode += 1
            recent_returns.append(ep_return)
            train_writer.writerow([step, episode, ep_return, ep_len, round(epsilon, 4)])
            train_log.flush()
            if episode % 5 == 0:
                fps = step / (time.time() - start)
                print(f"step {step:>8} | ep {episode:>4} | return {ep_return:6.1f} | "
                      f"mean(last 10) {np.mean(recent_returns[-10:]):6.1f} | "
                      f"eps {epsilon:.3f} | {fps:.0f} steps/s")
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        # Gradient step on a random minibatch from replay memory.
        if step > args.learning_starts and step % args.train_freq == 0:
            s, a, r, s2, d = buffer.sample(args.batch_size, device)
            with torch.no_grad():
                target_q = r + args.gamma * (1.0 - d) * target_net(s2).max(dim=1).values
            q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, target_q) if args.loss == "huber" else F.mse_loss(q, target_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        if use_target and step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % args.eval_every == 0 or step == args.total_steps:
            eval_return = evaluate(q_net, args.env_id, args.seed, args.eval_episodes,
                                   args.eval_epsilon, device)
            avg_q = average_max_q(q_net, holdout, device)
            eval_writer.writerow([step, eval_return, avg_q])
            eval_log.flush()
            print(f"[eval] step {step} | return {eval_return:.1f} | avg max Q {avg_q:.3f}")
            torch.save(q_net.state_dict(), os.path.join(args.out_dir, "last.pt"))
            if eval_return > best_eval:
                best_eval = eval_return
                torch.save(q_net.state_dict(), os.path.join(args.out_dir, "best.pt"))

    env.close()
    train_log.close()
    eval_log.close()
    print(f"done in {(time.time() - start) / 60:.1f} min, best eval return {best_eval:.1f}")


if __name__ == "__main__":
    main()
