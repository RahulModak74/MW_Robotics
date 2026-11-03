#!/usr/bin/env python3
"""
train_ppo_teacher_3d.py - stronger teacher for the 3D body: stable-baselines3 PPO on CPU (parallel envs).
CEM over a linear policy plateaus at ~90 steps on Humanoid-v5 (domain/29DoF/PROGRESS_LOG.md); PPO is the
conventional fix.  Its (s, a) rollouts are then fed to train_mw_3d.py --teacher ppo, which trains the MW
policy manifold (Pyro MWVAE) on top.  Saves checkpoints_3d/<env>_ppo{.zip,_vecnorm.pkl,_log.json}.

    python train_ppo_teacher_3d.py --env Humanoid-v5 --timesteps 10000000 --envs 10
"""
import os, sys, json, time, argparse
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_3d")
os.makedirs(CK, exist_ok=True)


class Prog(BaseCallback):
    def __init__(self, path, every=1_000_000):
        super().__init__(); self.path, self.every, self.t0, self.next, self.log = path, every, time.time(), every, []

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            eps = [e for e in self.model.ep_info_buffer]
            L = float(np.mean([e["l"] for e in eps])) if eps else 0.0
            R = float(np.mean([e["r"] for e in eps])) if eps else 0.0
            self.log.append({"timesteps": self.num_timesteps, "ep_len": L, "ep_rew": R, "sec": time.time() - self.t0})
            print(f"[{self.num_timesteps/1e6:5.1f}M] ep_len={L:6.1f} ep_rew={R:8.1f} ({time.time()-self.t0:.0f}s)", flush=True)
            self.model.save(self.path); self.training_env.save(self.path + "_vecnorm.pkl")
            json.dump(self.log, open(self.path + "_log.json", "w"), indent=1)
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Humanoid-v5"); ap.add_argument("--timesteps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(2)
    tag = a.env.split("-")[0].lower(); path = os.path.join(CK, f"{tag}_ppo")
    venv = VecNormalize(make_vec_env(a.env, n_envs=a.envs, vec_env_cls=SubprocVecEnv), gamma=0.99)
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0), device="cpu", verbose=0, seed=0)
    model.learn(a.timesteps, callback=Prog(path)); model.save(path); venv.save(path + "_vecnorm.pkl")
    print("done", flush=True)
