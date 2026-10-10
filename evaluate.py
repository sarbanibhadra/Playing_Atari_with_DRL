"""
Play a trained DQN agent, report its scores and save a GIF of one episode.

Usage:
  python evaluate.py --checkpoint runs/pong/best.pt --episodes 5 --gif pong_agent.gif
"""

import argparse

import imageio.v2 as imageio
import numpy as np
import torch

from dqn_pong import QNetwork, make_env, pick_device, select_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-id", default="ALE/Pong-v5")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--gif", default="", help="save the first episode as a GIF")
    p.add_argument("--max-gif-frames", type=int, default=3000)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    env = make_env(args.env_id, args.seed, render_mode="rgb_array" if args.gif else None)
    q_net = QNetwork(env.action_space.n).to(device)
    q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    q_net.eval()

    returns, frames = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            if args.gif and ep == 0 and len(frames) < args.max_gif_frames:
                frames.append(env.render())
            action = select_action(q_net, obs, args.epsilon, env.action_space.n, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        print(f"episode {ep + 1}: return {total:.1f}")
    env.close()

    print(f"mean return over {args.episodes} episodes: {np.mean(returns):.2f} "
          f"(std {np.std(returns):.2f})")

    if args.gif and frames:
        # Keep every second frame to limit file size; 15 fps playback.
        imageio.mimsave(args.gif, frames[::2], duration=1 / 15, loop=0)
        print(f"saved {args.gif} ({len(frames[::2])} frames)")


if __name__ == "__main__":
    main()
