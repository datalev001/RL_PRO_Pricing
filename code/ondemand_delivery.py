import pandas as pd, numpy as np

# load & basic profile ---------------------------------------------------
df = pd.read_csv("on-demand_delivery.csv",
                 parse_dates=["timestamp"])

print(df.shape)                       # (92 318, 11)
print(df.isna().mean().round(3))      # no missing values

# surge-switch heatmap ---------------------------------------------------
surge_shift = (df.groupby("zone_id")["surcharge_$"]
                 .apply(lambda s: s.diff().abs())
                 .rename("abs_jump"))
print("Pct of big jumps (≥4 $):",
      (surge_shift >= 4).mean().round(3))

# reward components ------------------------------------------------------
df["late_penalty"] = np.clip(df["avg_delivery_min"] - 30, 0, None) * 0.5
df["reward"]       = df["gross_revenue"] - df["rider_cost"] - df["late_penalty"]

print(df[["reward", "surcharge_$"]].groupby("surcharge_$")
        .agg(["mean", "std"]).round(2))

# volatility around rain & peaks ----------------------------------------
df["rain_bin"] = pd.qcut(df["rain_mm"], 4, labels=["dry", "light", "mod", "heavy"])
pivot = (df.pivot_table(index="is_peak", columns="rain_bin",
                        values="reward", aggfunc="std")
           .round(1))
print(pivot)

##############
"""
On-Demand Delivery Surcharge – PPO vs. Vanilla Actor–Critic
-----------------------------------------------------------
• Reads the synthetic log  on-demand_delivery.csv
• Builds a minimal 15-minute RL environment
• Trains two agents:
      1) PPO  (clip trust-region)
      2) A2C  (baseline Actor–Critic, no clip)
• Prints key metrics and plots learning curves side by side
-----------------------------------------------------------
Install once:
pip install gymnasium stable-baselines3 torch matplotlib pandas numpy
"""

import os, math, random, numpy as np, pandas as pd, matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

DATA_CSV = "on-demand_delivery.csv"
assert os.path.exists(DATA_CSV), "CSV not found – run the data-generation script first"
df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
print("✅ log loaded:", df.shape)

# ----------------------------------------------------------------------
# Gym env
# ----------------------------------------------------------------------
SURGE_LEVELS = np.array([0, 2, 4, 6])           # action = index 0-3
LATE_PENALTY = 0.5                              # ¥ per minute above 30

class DeliveryEnv(gym.Env):
    """
    One step = one 15-min slot for a single zone.
    Observation  (4 dims) : pending, riders, rain, is_peak   (z-score)
    Action       (Disc 4) : index of surcharge tier          {0,1,2,3}
    Reward                 revenue – rider_cost – late_penalty
    """
    def __init__(self, log_df: pd.DataFrame):
        super().__init__()
        self.df      = log_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        self.ptr     = 0
        # z-score scalers
        self.mu  = self.df[["pending_orders", "online_riders", "rain_mm"]].mean()
        self.sig = self.df[["pending_orders", "online_riders", "rain_mm"]].std()
        # gym spaces
        self.observation_space = spaces.Box(low=-5, high=5, shape=(4,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

    def _get_obs(self, row):
    # z-score the three numeric signals
        num = ((row[["pending_orders",
                     "online_riders",
                     "rain_mm"]] - self.mu) / self.sig).astype("float32").to_numpy()
        # fetch peak flag as float32, then concatenate
        peak_flag = np.array([row["is_peak"]], dtype=np.float32)
        return np.concatenate([num, peak_flag])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = 0
        row = self.df.loc[self.ptr]
        return self._get_obs(row), {}

    def step(self, action: int):
        row = self.df.loc[self.ptr]
        # replace historical surcharge by agent decision
        surcharge = SURGE_LEVELS[action]
        delivered = min(row.pending_orders, row.online_riders)
        # basic late time approximation
        wait_min  = max(10, 30 + (row.pending_orders - row.online_riders)*0.8 - surcharge*1.2)
        late_pen  = max(0, wait_min - 30) * LATE_PENALTY
        revenue   = delivered * (18 + surcharge)
        rider_cost= row.online_riders*12 + delivered*surcharge*0.5
        reward    = revenue - rider_cost - late_pen

        self.ptr += 1
        done = self.ptr >= len(self.df)-1
        next_obs = self._get_obs(self.df.loc[self.ptr]) if not done else np.zeros(4, np.float32)
        info = dict(revenue=revenue, late_pen=late_pen)
        return next_obs, reward, done, False, info

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
class RewardCallback(BaseCallback):
    """Stores smoothed reward for plotting."""
    def __init__(self, window=500):
        super().__init__()
        self.window = window
        self.rewards = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0 and "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            self.rewards.append(ep_reward)
        return True

def make_env():
    return Monitor(DeliveryEnv(df))

vec_env = DummyVecEnv([make_env])

# ----------------------------------------------------------------------
# Train PPO
# ----------------------------------------------------------------------
ppo_cb = RewardCallback()
ppo = PPO("MlpPolicy", vec_env, learning_rate=3e-4,
          n_steps=2048, batch_size=512, gamma=0.99,
          clip_range=0.2, ent_coef=0.01, verbose=0)
print("⏳  Training PPO ...")
ppo.learn(total_timesteps=60_000, callback=ppo_cb)

# ----------------------------------------------------------------------
# 4.  Train vanilla Actor–Critic (A2C)
# ----------------------------------------------------------------------
a2c_cb = RewardCallback()
a2c = A2C("MlpPolicy", vec_env, learning_rate=2e-4,
          n_steps=5, gamma=0.99, ent_coef=0.0, verbose=0)
print("⏳  Training A2C ...")
a2c.learn(total_timesteps=60_000, callback=a2c_cb)

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(agent, n_steps=20_000):
    test_env = DeliveryEnv(df.sample(frac=1.0, random_state=123).reset_index(drop=True))
    obs, _ = test_env.reset()
    rewards = []
    for _ in range(n_steps):
        act, _ = agent.predict(obs, deterministic=True)
        obs, r, done, _, _ = test_env.step(int(act))
        rewards.append(r)
        if done: break
    return np.mean(rewards), np.std(rewards)

ppo_mean, ppo_std = evaluate(ppo)
a2c_mean, a2c_std = evaluate(a2c)

print("\n=== Final offline evaluation (20k steps) ===")
print(f"PPO : mean reward = {ppo_mean:8.2f} ± {ppo_std:5.2f}")
print(f"A2C : mean reward = {a2c_mean:8.2f} ± {a2c_std:5.2f}")

# ----------------------------------------------------------------------
# Plot learning curves
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 3))
plt.plot(pd.Series(ppo_cb.rewards).rolling(20).mean(), label="PPO (smoothed)")
plt.plot(pd.Series(a2c_cb.rewards).rolling(20).mean(), label="A2C (smoothed)")
plt.title("Training reward comparison")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.legend(); plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# Deeper evaluation
# ----------------------------------------------------------------------
def rollout(agent, n_steps=10_000):
    env = DeliveryEnv(df.sample(frac=1.0, random_state=999).reset_index(drop=True))
    obs, _ = env.reset()
    stats = {"reward": [], "revenue": [], "late_pen": [], "surcharge": []}
    for _ in range(n_steps):
        act, _ = agent.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(int(act))
        stats["reward"].append(r)
        stats["revenue"].append(info["revenue"])
        stats["late_pen"].append(info["late_pen"])
        stats["surcharge"].append(SURGE_LEVELS[int(act)])
        if done:
            break
    return pd.DataFrame(stats)

ppo_roll = rollout(ppo)
a2c_roll = rollout(a2c)

def kpi(df_roll, name):
    print(f"\n{name}  -  {len(df_roll)} steps")
    print(" mean   reward :", f"{df_roll.reward.mean():8.2f}")
    print(" std    reward :", f"{df_roll.reward.std():8.2f}")
    print(" mean   revenue:", f"{df_roll.revenue.mean():8.2f}")
    print(" mean late_pen :", f"{df_roll.late_pen.mean():8.2f}")
    for lvl in SURGE_LEVELS:
        pct = (df_roll.surcharge == lvl).mean()*100
        print(f"   surcharge {lvl:>2} ¥ chosen {pct:5.1f} %")

kpi(ppo_roll, "PPO")
kpi(a2c_roll, "A2C")

# ----------------------------------------------------------------------
# Three-panel visual comparison
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# learning curves (already collected)
axes[0].plot(pd.Series(ppo_cb.rewards).rolling(20).mean(), label="PPO")
axes[0].plot(pd.Series(a2c_cb.rewards).rolling(20).mean(), label="A2C")
axes[0].set_title("Moving-average episode reward")
axes[0].set_xlabel("Episode"); axes[0].set_ylabel("R"); axes[0].legend()

# reward histogram
axes[1].hist(ppo_roll.reward, bins=40, alpha=.6, label="PPO")
axes[1].hist(a2c_roll.reward, bins=40, alpha=.6, label="A2C")
axes[1].set_title("Reward distribution (10k steps)")
axes[1].set_xlabel("reward"); axes[1].legend()

# action frequency
bar_w = 0.35
x_pos = np.arange(len(SURGE_LEVELS))
axes[2].bar(x_pos-bar_w/2,
            [ (ppo_roll.surcharge==lvl).mean() for lvl in SURGE_LEVELS ],
            width=bar_w, label="PPO")
axes[2].bar(x_pos+bar_w/2,
            [ (a2c_roll.surcharge==lvl).mean() for lvl in SURGE_LEVELS ],
            width=bar_w, label="A2C")
axes[2].set_xticks(x_pos); axes[2].set_xticklabels(SURGE_LEVELS)
axes[2].set_title("Chosen surcharge share")
axes[2].set_xlabel("tier ($)"); axes[2].legend()

plt.tight_layout(); plt.show()