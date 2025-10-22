import env.custom_env 
import traceback  # 匯入 traceback 模組以便印出詳細錯誤
import seaborn as sns
import csv
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
from typing import Union
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Set, Optional
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
import cloudpickle, os
from typing import List, Tuple, Dict, Any
import math, torch
import copy, torch
from typing import Tuple
import torch.nn as nn
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict
import torch
import os
import gym
import numpy as np
import time
import json
import random
   # 确保 custom_env.py 中的 register() 被执行
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CheckpointCallback,
    BaseCallback
)
from multiprocessing import Pool, cpu_count
from tqdm import tqdm # 用於顯示進度條，更友善
import pandas as pd
from stable_baselines3.common.logger import configure 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 修改：make_env 加一个 difficulty 参数
class CurriculumCallback(BaseCallback):
    """
    定期將 DummyVecEnv 的難度，優先複習表現較差的難度，並自動複習 hardseed。
    複習結束後自動切回最高難度，並自動測試清理 hardseed。
    支援 hardseed reward 動態更新 + 機率式複習 seed
    """
    def __init__(self, difficulty_list, eval_env, hardseeds_path: str = "./logs/hard_seeds.json",
                 switch_freq=50000, review_steps=10000,
                 hardseed_reward_threshold=100, hardseed_n_eval=2,
                 verbose=1, shared_flags=None, cooldown_steps=30000,seeds_per_group=5):
        super().__init__(verbose)
        self.shared_flags = shared_flags or {}
        self.difficulty_list = difficulty_list
        self.eval_env = eval_env
        # hard_seeds 應為 {難度: {seed: reward, ...}, ...}
        self.hardseed_path = hardseeds_path
        with open(self.hardseed_path, "r") as f:
            # 檔案格式：{ "0.50": {"seed1": 123.4, "seed2": 56.7}, ... }
            self.hard_seeds = json.load(f)
        self.switch_freq = switch_freq
        self.review_steps = review_steps
        self.last_switch = 0
        self.reviewing = False
        self.review_countdown = 0
        self.bad_hardseeds = []
        self.current_seed_idx = 0
        self.review_step_count = 0
        self.prev_difficulty = None
        self.hardseed_reward_threshold = hardseed_reward_threshold
        self.hardseed_n_eval = hardseed_n_eval
        self.cooldown_steps = cooldown_steps
        self.next_allowed_switch = 0  # 下一次允許觸發複習的步數
        self.seeds_per_group = seeds_per_group
    def _sample_seeds_across_all_diffs(self) -> list[int]:
        bucket1, bucket2, bucket3, bucket4 = [], [], [], []   # 200-250, 125-200, 0-120, <0
        for diff_dict in self.hard_seeds.values():
            for seed_str, reward in diff_dict.items():
                r = float(reward)
                if 200 <= r < 250:
                    bucket1.append(int(seed_str))
                elif 125 <= r < 200:
                    bucket2.append(int(seed_str))
                elif 0   <= r < 120:
                    bucket3.append(int(seed_str))
                elif r < 0:
                    bucket4.append(int(seed_str))

        sampled = []
        for bucket in (bucket1, bucket2, bucket3, bucket4):
            if bucket:
                sampled += random.sample(bucket, min(self.seeds_per_group, len(bucket)))
        return sampled

    def _on_step(self) -> bool:
        # ───────────────────────── 冷卻中 ─────────────────────────
        if self.n_calls < self.next_allowed_switch:
            return True

        # ───────────────   正在 reviewing 模式   ────────────────
        if self.reviewing:
            self.shared_flags["reviewing"] = True
            steps_per_seed = max(1, self.review_steps // len(self.bad_hardseeds))
            seed = self.bad_hardseeds[self.current_seed_idx]
            self.training_env.env_method("set_seed", seed)
            self.review_step_count += 1

            if self.review_step_count >= steps_per_seed:
                self.current_seed_idx += 1
                self.review_step_count = 0

                if self.current_seed_idx >= len(self.bad_hardseeds):
                    self._test_and_remove_badseeds()
                    print(f"[HardSeedReview] 複習結束，切回難度 {self.prev_difficulty:.2f}")
                    self.training_env.env_method("set_difficulty", self.prev_difficulty)
                    self.training_env.env_method("set_seed", None)
                    self.reviewing = False
                    self.shared_flags["reviewing"] = False
                    self.last_switch = self.n_calls
            return True

        # ──────────────────   是否到時間啟動複習？   ────────────────
        if (self.n_calls - self.last_switch) <= self.switch_freq:
            return True

        # ============  (A) 準備進入 hard-seed 複習  =============
        self.prev_difficulty = self.training_env.env_method("get_difficulty")[0]
        sampled_seeds = self._sample_seeds_across_all_diffs()     # ←★ 新抽樣方式
        if not sampled_seeds:
            print("[HardSeedReview] 找不到任何待複習的 hard-seed，跳過。")
            self.last_switch = self.n_calls
            self.next_allowed_switch = self.n_calls + self.cooldown_steps
            with open(self.hardseed_path, "w") as f:          # ★
                json.dump(self.hard_seeds, f, ensure_ascii=False, indent=2)
            return True

        # ============  (B) 對抽到的 seed 先離線評估一次  ==========
        bad_seeds = []
        for seed in sampled_seeds:
            avg_reward = self._evaluate_seed(seed)

            # 把最新 reward 寫回檔案結構
            for diff in self.hard_seeds:
                if str(seed) in self.hard_seeds[diff]:
                    self.hard_seeds[diff][str(seed)] = avg_reward

            if avg_reward >= 250:
                for diff in self.hard_seeds:
                    self.hard_seeds[diff].pop(str(seed), None)
                print(f"[HardSeedReview] seed {seed} 已過關，自動移除。")
            elif avg_reward < self.hardseed_reward_threshold:
                bad_seeds.append(seed)
                print(f"[HardSeedReview] seed {seed} 沒過關，加入 badseed。")

        # ============  (C) 真正進入 reviewing  ============
        if bad_seeds:
            self.shared_flags["reviewing"] = True  
            print(f"\n[HardSeedReview] 進入複習，共 {len(bad_seeds)} 個 seed\n")
            self.bad_hardseeds = bad_seeds
            self.current_seed_idx = 0
            self.review_step_count = 0
            self.reviewing = True
            self.training_env.env_method("set_seed", bad_seeds[0])
        else:
            print("[HardSeedReview] 沒有需要複習的 hard-seed，本次跳過。")

        self.last_switch        = self.n_calls
        self.next_allowed_switch = self.n_calls + self.cooldown_steps
        return True

    def _evaluate_seed(self, seed):
        self.eval_env.unwrapped.set_seed(seed)
        total_rewards = []
        for _ in range(self.hardseed_n_eval):
            obs, _ = self.eval_env.reset()
            done = False
            total = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total += reward
            total_rewards.append(total)
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"[HardSeedTest] seed {seed}, 平均回報 {avg_reward:.2f}")
        return avg_reward

    def _test_and_remove_badseeds(self):
        """複習後再測試 bad_hardseeds，會的自動移除"""
        to_remove = []
        for seed in self.bad_hardseeds:
            avg_reward = self._evaluate_seed(seed)
            # 寫回最新 reward
            for d in self.hard_seeds:
                if str(seed) in self.hard_seeds[d]:
                    self.hard_seeds[d][str(seed)] = avg_reward
            if avg_reward >= self.hardseed_reward_threshold:
                for d in self.hard_seeds:
                    if str(seed) in self.hard_seeds[d]:
                        self.hard_seeds[d].pop(str(seed))
                        print(f"[HardSeed] seed {seed} 學會了，已移除。")
                to_remove.append(seed)
        self.bad_hardseeds = [s for s in self.bad_hardseeds if s not in to_remove]
        with open(self.hardseed_path, "w") as f:
            json.dump(self.hard_seeds, f, ensure_ascii=False, indent=2)
class AutoDifficultyCallback(BaseCallback):
    def __init__(self, eval_env, hard_seeds, eval_freq=10000, reward_threshold=250, increase=0.05, verbose=1,
                 difficulty_list=None, shared_flags=None, cooldown_steps=30000, hardseed_save_path="./logs/hard_seeds.json"):
        super().__init__(verbose)
        self.shared_flags = shared_flags or {}
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.reward_threshold = reward_threshold
        self.increase = increase
        self.current_difficulty = eval_env.unwrapped.difficulty
        # {難度: {seed: reward}}
        if hard_seeds is None:
            with open(hardseed_save_path, "r") as f:
                self.hard_seeds = json.load(f)
        else:
            self.hard_seeds = hard_seeds
        self.hardseed_save_path = hardseed_save_path

        self.difficulty_list = list(self.hard_seeds.keys())
        self.cooldown_steps = cooldown_steps
        self.next_allowed_upgrade = 0
        

    def save_hard_seeds(self):
        # 存成 {難度: {seed: reward}}
        import json
        with open(self.hardseed_save_path, "w") as f:
            json.dump(self.hard_seeds, f, ensure_ascii=False, indent=2)
        print(f"[AutoDiff] 已保存 hard_seeds 至 {self.hardseed_save_path}")

    def _on_step(self) -> bool:
        if self.n_calls < self.next_allowed_upgrade or self.shared_flags.get("reviewing", False):
            return True
        # 每 eval_freq 步做一次評估
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            seeds = []
            for _ in range(100):
                obs, info  = self.eval_env.reset()
                seed = info.get("seed", None)
                seeds.append(seed)
                done = False
                total = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    total += reward
                rewards.append(total)
            avg_reward = sum(rewards) / len(rewards)
            print(f"[AutoDifficulty] 當前難度 {self.current_difficulty:.2f}，平均回報 {avg_reward:.2f}")
            self.logger.record("curriculum/difficulty", self.current_difficulty)

            if avg_reward > 150:
                print(f"[AutoDiff] 收集 hard seed ）")
                key = str(round(self.current_difficulty, 2))  # 用字串當 key 方便 json 存
                if key not in self.hard_seeds:
                    self.hard_seeds[key] = {}
                for seed, total in zip(seeds, rewards): 
                    if seed is None:
                        print(f"[AutoDiff] 沒有hard seed （回報 {total:.1f}）")
                        continue
                    # 只收錄低於平均值 * 0.5 的 seed，並存 reward
                    if total < avg_reward * 0.5:
                        # 若已經存過該 seed，保留最差 reward（可根據需要改成保存平均）
                        prev = self.hard_seeds[key].get(str(seed), None)
                        if prev is None or total < prev:
                            print(f"[AutoDiff] 收集 hard seed {seed}（回報 {total:.1f}）")
                            self.hard_seeds[key][str(seed)] = total
                self.save_hard_seeds()
            # 超過門檻自動升級
            if avg_reward > self.reward_threshold and self.current_difficulty < 1.0:
                self.current_difficulty = min(self.current_difficulty + self.increase, 1.0)
                self.training_env.env_method('set_difficulty', self.current_difficulty)
                self.eval_env.unwrapped.difficulty = self.current_difficulty
                print(f"[AutoDifficulty] 難度升級為 {self.current_difficulty:.2f}")
                self.next_allowed_upgrade = self.n_calls + self.cooldown_steps  # 設冷卻
        return True
class CustomStopCallback(BaseCallback):
    def __init__(self, eval_env, reward_threshold=300, target_difficulty=0.7, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.reward_threshold = reward_threshold
        self.target_difficulty = target_difficulty
        self.stop_training = False

    def _on_step(self) -> bool:
        # 只有難度達標才評估 reward
        current_difficulty = self.eval_env.unwrapped.difficulty
        if current_difficulty >= self.target_difficulty:
            rewards = []
            for _ in range(10):
                obs, _ = self.eval_env.reset()
                done = False
                total = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    total += reward
                rewards.append(total)
            avg_reward = sum(rewards) / len(rewards)
            print(f"[CustomStop] 難度 {current_difficulty:.2f}, 平均回報 {avg_reward:.2f}")
            if avg_reward > self.reward_threshold:
                print("[CustomStop] 達到最高難度且平均回報達標，訓練結束！")
                return False 
        return True
from functools import partial

def make_one_env(env_id: str, rank: int, difficulty: float, seed: int):
    env = gym.make(env_id, difficulty=difficulty)
    env.reset(seed=seed + rank)
    return env

def make_vec_env(env_id, num_envs, difficulty, base_seed, use_subproc=True):
    thunks = [partial(make_one_env, env_id, i, difficulty, base_seed) for i in range(num_envs)]
    return SubprocVecEnv(thunks) if use_subproc else DummyVecEnv(thunks)

def train(config: Dict[str, Any]):
    if torch.cuda.is_available():
        # 建議三個進程就設 0.3 左右（30% * 3 ≈ 90%），避免互相擠爆
        torch.cuda.set_per_process_memory_fraction(0.25, device=0)
    # 修改：改用 custom 环境 ID
    run_id            = config["run_id"]
    env_id = config.get("env_id","BipedalWalkerCustom-v0")
    num_cpu = int(config.get("num_envs",8))
    total_timesteps = int(config.get("total_timestepts",100_000_000_0))
    log_dir = os.path.join("./logs", f"run_{run_id}")
    best_model_dir = os.path.join("./best_model", f"run_{run_id}")
    difficulty_list = [round(0.05 * i, 2) for i in range(1, 21)]  # [0.1,0.2,...,1.0]
    # 你想要的难度（0.0 全平地，1.0 原生 Hardcore） 
    difficulty = 0.0
    base_seed         = int(config.get("base_seed", 42))
    device            = config.get("device", "cuda")  # 同一張卡同時跑多個進程會搶資源，視顯存調整
    use_subproc       = bool(config.get("use_subproc", True))
    # 建立多环境並加上 Monitor
    random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    vec_env=make_vec_env(env_id,num_envs=num_cpu,difficulty=difficulty,base_seed=base_seed,use_subproc=use_subproc)
    vec_env = VecMonitor(vec_env, log_dir)

    # 設定 SB3 logger
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    ckpt_to_resume = config.get("resume_path", None)
    if ckpt_to_resume and os.path.exists(ckpt_to_resume):
        print(f"[run {run_id}] 🔄 載入舊模型繼續訓練：{ckpt_to_resume}")
        model = PPO.load(ckpt_to_resume, env=vec_env, device=device)
        model.set_env(vec_env)
    else:
        print(f"run {run_id}重新開始新模型")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda",
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
        )
        model.set_logger(new_logger)

    # 評估用環境，也要传 difficulty
    eval_env = gym.make(env_id, difficulty=difficulty)
    difficulty_list = [round(0.05 * i, 2) for i in range(1, 21)]
    hard_seeds = {d: [] for d in difficulty_list}
    shared_flags = {"reviewing": False}
    curriculum_callback = CurriculumCallback(difficulty_list,eval_env, hardseeds_path="./logs/hard_seeds.json",switch_freq=200000, review_steps=60000,hardseed_reward_threshold=250, hardseed_n_eval=10,verbose=1,shared_flags=shared_flags,cooldown_steps=30000)
    custom_stop_callback = CustomStopCallback(eval_env, reward_threshold=300, target_difficulty=1.0, verbose=1)
    eval_callback = EvalCallback(
    eval_env,
     callback_on_new_best=custom_stop_callback,
    best_model_save_path=best_model_dir,  # 儲存目錄
    log_path=log_dir,                     # 評估 log
    eval_freq=10000,                      # 每多少步做一次評估
    n_eval_episodes=5,                    # 每次評估用幾次 episode
    deterministic=True,
    render=False
)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=best_model_dir,
        name_prefix="ppo_checkpoint-hard1"
    )
    auto_difficulty_callback = AutoDifficultyCallback(
        eval_env,None, eval_freq=20_000, reward_threshold=250, increase=0.05, verbose=1,shared_flags=shared_flags,cooldown_steps=0,hardseed_save_path="./logs/hard_seeds.json"
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[ curriculum_callback,auto_difficulty_callback,checkpoint_callback,eval_callback],
        tb_log_name=f"ppo_bipedalwalker_custom-{run_id}"
    )

    model.save(os.path.join(best_model_dir, f"ppo_bipedalwalker_custom_final{run_id}"))
    print(f"[run {run_id}] ✅ 訓練完成，已儲存：")
def more_train():
    mp.set_start_method("spawn",force=True)
    NUM_RUNS=4
    TOTAL_TIMESTEPTS =500_000_000_0
    base_configs =[]
    for run_id in range(9,NUM_RUNS +10):
        base_configs.append(dict(
            run_id=run_id,
            env_id="BipedalWalkerCustom-v0",
            num_envs=8,                 # 依 CPU 調整；你之前寫 256 很吃 RAM 與 CPU，建議先 16~64
            total_timesteps= TOTAL_TIMESTEPTS,
            difficulty=0.0 ,  # 範例：不同 run 用不同起始難度
            base_seed=random.randint(0, 2**31-1) * run_id, # 每個 run 不同 seed
            learning_rate=2.5e-4,
            n_steps=1024,                 # 多進程並跑建議比單跑小一點，避免顯存爆
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            use_subproc=True,
            #resume_path 可填入各自 run 的舊 checkpoint
            #resume_path= "./best_model/run_1/ppo_checkpoint_run1_8000000_steps.zip"
        ))
    max_workers = min(NUM_RUNS,os.cpu_count()or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(train, base_configs))

def test():
    diff=1.0
    for i in range (10):
    # 測試也要用 custom 
        test_seed = 223913247
        #test_seed = 1451341367
        print(diff)
        # env =make_vec_env("BipedalWalkerCustom-v0",num_envs=1, difficulty=diff,base_seed=test_seed,use_subproc=False)
        env = gym.make("BipedalWalkerCustom-v0", difficulty=diff, render_mode="human")
        # model = PPO.load("./best_model/model2.zip",env=env)
        with open("./models/child_model2_x_model3_fused_full.pkl", "rb") as f:
            model = cloudpickle.load(f)
        # model.set_env(vec_env)
        #print(model.policy)          # 一行就能看到主要子模組
        #inspect_crosstalk_weights(model.policy)
        # print("\n*** 正在執行消融操作：將交叉通道權重歸零... ***")
        # zero_initialize_crosstalk(model.policy)
        # print("*** 消融操作完成！模型現在只依靠純淨通道工作。 ***")
        
        # # --- 再次檢查權重，確認已歸零 (可選) ---
        # print("\n--- 檢查【歸零後】的交叉通道權重 ---")
        # inspect_crosstalk_weights(model.policy)
        obs, info = env.reset(seed=test_seed)
        # obs, info = env.reset()
        print("info =", info)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    
    print(f"✅ 測試總回報：{total_reward:.2f}")
    
    env.close()
def inspect_crosstalk_weights(policy: nn.Module):
    """
    檢查並打印模型中交叉通道權重的學習情況。
    """
    print("\n--- 交叉通道權重學習情況審查 ---")
    
    total_cross_weights = 0
    learned_cross_weights = 0
    
    with torch.no_grad():
        for name, module in policy.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                in_half = W.shape[1] // 2
                out_half = W.shape[0] // 2

                if in_half == 0 or out_half == 0:
                    continue

                # 選中右上角 (母 -> 父) 和 左下角 (父 -> 母) 區塊
                cross_mf = W[:out_half, in_half:]
                cross_fm = W[out_half:, :in_half]

                # 計算這兩個區塊權重的平均絕對值
                mean_abs_mf = torch.abs(cross_mf).mean()
                mean_abs_fm = torch.abs(cross_fm).mean()
                
                print(f"層 '{name}':")
                print(f"  - 母->父 通道權重平均絕對值: {mean_abs_mf:.8f}")
                print(f"  - 父->母 通道權重平均絕對值: {mean_abs_fm:.8f}")

                # 統計非零權重數量
                total_cross_weights += cross_mf.numel() + cross_fm.numel()
                learned_cross_weights += torch.count_nonzero(cross_mf) + torch.count_nonzero(cross_fm)

    if total_cross_weights > 0:
        learning_ratio = (learned_cross_weights / total_cross_weights) * 100
        print(f"\n總結：共有 {learned_cross_weights} / {total_cross_weights} ({learning_ratio:.2f}%) 的交叉權重已從0開始學習。")
    else:
        print("\n總結：模型中沒有可檢查的交叉通道。")
def prune_hardseeds():
    import json
    # 載入 hard_seeds
    hardseed_path="./logs/hard_seeds.json"
    model_path="./best_model/ppo_checkpoint-hard6_8000000_steps.zip"
    env_id="BipedalWalkerCustom-v0"
    eval_n=1
    reward_threshold=250
    difficulty_list=None
    with open(hardseed_path, "r") as f:
        data = json.load(f)

    # 2. 轉新格式（如果是 list 就轉成 {seed: 0}）
    new_data = {}
    for diff, seeds in data.items():
        if isinstance(seeds, list):
            new_data[diff] = {str(s): 0 for s in seeds}
        else:
            new_data[diff] = seeds

    # 3. 載入模型
    model = PPO.load(model_path)

    # 4. 依序對每個難度測試所有 seed
    pruned_data = {}
    diffs = difficulty_list or sorted([float(k) for k in new_data.keys()])
    for diff in diffs:
        diff_str = str(diff)
        seed_dict = new_data.get(diff_str, {})
        if not seed_dict:
            continue
        print(f"\n==== 難度 {diff_str} ====")
        env = gym.make(env_id, difficulty=diff)
        new_seed_dict = {}
        for seed_str in list(seed_dict.keys()):
            rewards = []
            for _ in range(eval_n):
                seedtest=int(seed_str)
                obs, info = env.reset(seed=seedtest)
                print("info =", info)
                done = False
                total_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                rewards.append(total_reward)
            avg_reward = sum(rewards) / len(rewards)
            print(f"Seed {seed_str} 平均回報：{avg_reward:.2f}", end=" ")
            # 只保留沒過關的 seed
            if avg_reward <= reward_threshold:
                new_seed_dict[seed_str] = avg_reward
                print("[保留]")
            else:
                print("[移除]")
        if new_seed_dict:
            pruned_data[diff_str] = new_seed_dict

    # 5. 覆寫 hard_seeds.json
    with open(hardseed_path, "w") as f:
        json.dump(pruned_data, f, indent=2)
    print("\n已完成所有 seed 測試，hard_seeds.json 已更新！")
def rename_modelname():
    model_dir = "./best_model/"
    files = sorted([f for f in os.listdir(model_dir) if f.endswith(".zip")])

    for i, filename in enumerate(files):
        new_name = f"{i}.zip"
        old_path = os.path.join(model_dir, filename)
        new_path = os.path.join(model_dir, new_name)
        os.rename(old_path, new_path)
        print(f"✅ 已將 {filename} ➜ {new_name}")

def sample_hardseeds(hardseed_path, num_samples=100, diff_min=0.1, diff_max=1.0):
    """ 從 hard_seeds.json 抽 (difficulty, seed) 並可限難度範圍 """
    with open(hardseed_path, "r") as f:
        hardseed_dict = json.load(f)

    all_pairs = []
    for diff_str, seeds in hardseed_dict.items():
        d = float(diff_str)
        if diff_min <= d <= diff_max:
            for seed_str in seeds.keys():
                all_pairs.append((d, int(seed_str)))
    return random.sample(all_pairs, min(num_samples, len(all_pairs)))
TEACHERS = None
DEVICE = None
def _init_worker(model_paths,device_str="cpu"):
    global TEACHERS,DEVICE
    DEVICE= torch.device(device_str)
    torch.set_grad_enabled(False)#全域地開關 Autograd（自動微分）功能，不執行反向傳播
    torch.set_num_threads(1)
    TEACHERS={os.path.basename(p):PPO.load(p,device=DEVICE)for p in model_paths}
@torch.inference_mode()
def _get_gaussian_params_step(model, obs_tensor):
    """
    單步：obs_tensor shape [1, obs_dim] => 回傳 mean[act_dim], log_std[act_dim]
    """
    policy = model.policy
    obs_tensor = obs_tensor.to(next(policy.parameters()).device)

    # 走標準路徑取 mean
    feats = policy.extract_features(obs_tensor)
    if policy.share_features_extractor:
        latent_pi, _ = policy.mlp_extractor(feats)
    else:
        latent_pi, _ = policy.mlp_extractor(feats, None)
    mean = policy.action_net(latent_pi)  # [1, act_dim]

    # 取得 log_std
    if hasattr(policy, "log_std") and policy.log_std is not None:
        log_std = policy.log_std.expand_as(mean)
    else:
        dist = policy.get_distribution(obs_tensor)
        if hasattr(dist, "log_std") and dist.log_std is not None:
            log_std = dist.log_std
        elif hasattr(dist, "distribution"):
            std = dist.distribution.stddev
            log_std = torch.log(std + 1e-8)
        else:
            raise RuntimeError("找不到 log_std，請檢查 SB3 版本")

    return mean.squeeze(0), log_std.squeeze(0)


@torch.inference_mode()
def _get_gaussian_params_batch(model, obs_batch):
    """
    批量：obs_batch shape [T, obs_dim] => 回傳 mean[T, act_dim], log_std[T, act_dim]
    """
    policy = model.policy
    device = next(policy.parameters()).device
    X = obs_batch.to(device)

    # 先算 mean（自帶 batch 維）
    feats = policy.extract_features(X)
    if policy.share_features_extractor:
        latent_pi, latent_vl = policy.mlp_extractor(feats)
    else:
        latent_pi,latent_vl = policy.mlp_extractor(feats, None)
    mean = policy.action_net(latent_pi)  # [T, act_dim]
    value =policy.value_net(latent_vl)
    # 取 log_std：優先使用 policy.log_std；否則從 distribution 取
    if hasattr(policy, "log_std") and policy.log_std is not None:
        log_std = policy.log_std.expand_as(mean)  # [T, act_dim]
    else:
        dist = policy.get_distribution(X)  # 會重新抽 features，但穩妥
        if hasattr(dist, "log_std") and dist.log_std is not None:
            log_std = dist.log_std
        elif hasattr(dist, "distribution"):
            std = dist.distribution.stddev
            log_std = torch.log(std + 1e-8)
        else:
            raise RuntimeError("找不到 log_std，請檢查 SB3 版本")

    return mean, log_std,value


# ========== Stage-1：只算 fitness + 輕量存 states ==========
def run_fitness_task(args):
    """
    args: (controller_name, diff, seed_list, env_id, eval_n, reward_threshold, tmp_dir)
    回傳: (controller_name, fitness_dict, tmp_paths)
    """
    (ctrl_name, diff, seed_list,
     env_id, eval_n, reward_th, tmp_dir) = args

    ctrl = TEACHERS[ctrl_name]
    env  = gym.make(env_id, difficulty=diff)

    tot_r = eps = passed = 0
    tmp_paths = []

    for seed in seed_list:                    # ← 多個 seed
        for ep in range(eval_n):
            obs, _ = env.reset(seed=to_uint32(seed))
            done, ep_r = False, 0.0
            traj = []

            while not done:
                obs_t = (torch.as_tensor(obs, dtype=torch.float32)
                           .unsqueeze(0).to(DEVICE))
                mean, _ = _get_gaussian_params_step(ctrl, obs_t)
                action  = mean.detach().cpu().numpy()

                traj.append(np.array(obs, copy=True))
                obs, r, term, trunc, _ = env.step(action)
                ep_r += r; done = term or trunc

            tot_r += ep_r; eps += 1
            if ep_r >= reward_th: passed += 1

            arr = np.asarray(traj, dtype=np.float16)
            os.makedirs(tmp_dir, exist_ok=True)
            fn = os.path.join(
                tmp_dir,
                f"{ctrl_name}__d{diff:.2f}__s{seed}__pid{os.getpid()}__ep{ep}.npz"
            )
            np.savez_compressed(
                fn,
                states=arr,
                diff=np.float32(diff),
                seed=np.int64(seed)
            )
            tmp_paths.append(fn)

    env.close()
    return ctrl_name, {"total_reward": tot_r, "episodes": eps, "passed": passed}, tmp_paths

def to_uint32(x: int) -> int:
    """将任意 Python int 映射到 32-bit 无符号整型范围，适合传给 env.reset(seed=...)."""
    return int(x) & 0xFFFFFFFF
# ========== Stage-2：離線對 Top-K 批量標註（不重跑環境） ==========
def offline_label_task(args):
    """
    args: (states_path, top_teacher_names)
    讀回 states，對 top_teacher 做批量前向 => 產生 label 檔（.label.npz）
    回傳: label_path
    """
    states_path, top_teacher_names = args
    data = np.load(states_path)
    S = data["states"].astype(np.float32)            # [T, obs_dim]
    diff = float(data["diff"])
    seed = int(data["seed"])
    T = S.shape[0]

    obs_batch = torch.from_numpy(S)                  # [T, obs_dim]
    means_list, logstds_list, actions_list, confs_list,value_list = [], [], [], [],[]

    with torch.inference_mode():
    # 先一次跑完所有老師，收齊 mean/logstd/value
        tmp = []
        for name in top_teacher_names:
            m, ls, v = _get_gaussian_params_batch(TEACHERS[name], obs_batch)  # [T, A], [T, A], [T, 1]
            tmp.append((
                m.detach().cpu().numpy(),       # [T, A]
                ls.detach().cpu().numpy(),      # [T, A] (log_std)
                v.detach().cpu().numpy().squeeze(-1)  # [T]
            ))

        # ====== Epistemic：老師間均值一致性 ======
        means_stack = np.stack([x[0] for x in tmp], axis=0)   # [Tchr, T, A]
        means_stack = np.transpose(means_stack, (1, 0, 2))    # [T, Tchr, A]
        inter_teacher_var = np.var(means_stack, axis=1)       # [T, A]
        agree_conf = 1.0 / (np.mean(inter_teacher_var, axis=1) + 1e-8)  # [T]
        # 簡單的 5~95 去尾 + [0,1] 歸一化
        def minmax_norm(x, eps=1e-8):
            lo, hi = np.percentile(x, 5), np.percentile(x, 95)
            x = np.clip(x, lo, hi)
            x = (x - lo) / (hi - lo + eps)
            return np.clip(x, 0.0, 1.0)
        agree_conf = minmax_norm(agree_conf)  # [T] in [0,1]

        # ====== 依序產生每位老師的 conf，含 value 門控 ======
        alpha = 0.6  # aleatoric vs epistemic 幾何融合權重
        k     = 0.5  # value 門控斜率
        for (m_cpu, ls_cpu, v_cpu) in tmp:
            # Aleatoric：反方差（對角高斯）
            var_cpu = np.exp(2.0 * ls_cpu)                       # [T, A]
            alea_conf = 1.0 / (np.mean(var_cpu, axis=1) + 1e-8)  # [T]
            alea_conf = minmax_norm(alea_conf)

            # 幾何融合（aleatoric × epistemic）
            conf_cpu = np.exp(alpha * np.log(alea_conf + 1e-12) +
                            (1 - alpha) * np.log(agree_conf + 1e-12))  # [T]

            # ====== Value 門控（越高價值 → 權重越大；你也可改成反向）======
            # 先把 value 做穩定化：去尾 + z-score + sigmoid 門控
            v_clip_lo, v_clip_hi = np.percentile(v_cpu, 5), np.percentile(v_cpu, 95)
            v_clip = np.clip(v_cpu, v_clip_lo, v_clip_hi)
            v_mu, v_std = np.mean(v_clip), (np.std(v_clip) + 1e-8)
            v_z = (v_clip - v_mu) / v_std
            gate = 1.0 / (1.0 + np.exp(-k * v_z))  # [T] in (0,1)
            conf_cpu *= gate
            conf_cpu = minmax_norm(conf_cpu)

            # 你原本就要存的資料
            a_cpu = m_cpu.copy()               # 以 mean 作為動作標註
            means_list.append(m_cpu)
            logstds_list.append(ls_cpu)
            actions_list.append(a_cpu)
            value_list.append(v_cpu)           # 仍保存每位老師自己的 value
            confs_list.append(conf_cpu)

    # 轉為 [T, Tchr, ...] 方便後續 concat（沿 T 維拼）
    Tchr = len(top_teacher_names)
    M = np.stack(means_list, axis=0).transpose(1, 0, 2)     # [T, Tchr, act_dim]
    LS = np.stack(logstds_list, axis=0).transpose(1, 0, 2)  # [T, Tchr, act_dim]
    A = np.stack(actions_list, axis=0).transpose(1, 0, 2)   # [T, Tchr, act_dim]
    C = np.stack(confs_list, axis=0).transpose(1, 0)        # [T, Tchr]
    V = np.stack(value_list, axis=0).transpose(1, 0)
    out = states_path.replace(".npz", ".label.npz")
    np.savez_compressed(out,
                        teacher_means=M.astype(np.float32),
                        teacher_logstds=LS.astype(np.float32),
                        teacher_actions=A.astype(np.float32),
                        confidences=C.astype(np.float32),
                        teacher_values=V.astype(np.float32),
                        diff=np.float32(diff),
                        seed=np.int64(seed))
    return out


# ========== 主流程：先 fitness 後收集 Top-K ==========
def test_population(
    model_paths,
    hardseed_path,
    env_id="BipedalWalkerCustom-v0",
    num_samples=300,
    eval_n_stage1=1,
    eval_n_stage2=10,
    reward_threshold=250,
    top_k=30,
    save_dir="./logs/ga_eval",
    save_name="mtkd_continuous.pt",
    num_workers=12,
    device_str="cpu",
    diff_min=0.0,
    diff_max=1.0,
):
    os.makedirs(save_dir, exist_ok=True)
    tmp_dir = os.path.join(save_dir, "tmp_states"); os.makedirs(tmp_dir, exist_ok=True)

    teacher_names = [os.path.basename(p) for p in model_paths]

    # 取樣 (diff, seed) ------------- 第一層批量切段 ----------------
    PAIR_BATCH = 50                      # 一段最多 100 對 (diff, seed)
    sampled_pairs = sample_hardseeds(hardseed_path, num_samples,
                                     diff_min=diff_min, diff_max=diff_max)
    sampled_pairs.sort()                   # optional: 讓同 diff 靠近

    pair_batches = [
        sampled_pairs[i : i + PAIR_BATCH]
        for i in range(0, len(sampled_pairs), PAIR_BATCH)
    ]
    print(f"Total (diff, seed) pairs = {len(sampled_pairs)}, "
          f"{len(pair_batches)} batches × {PAIR_BATCH} (<=) each")

    # 累積器
    fitness = {n: {"total_reward":0., "episodes":0, "passed":0} for n in teacher_names}
    paths_map = {n: [] for n in teacher_names}

    # --------- 逐段處理 Stage-1 (fitness) ----------
    SEED_BATCH = 10                        # 一個 Pool task 跑 10 seeds
    for b_idx, pair_batch in enumerate(pair_batches, 1):
        print(f"\n🔹 Batch {b_idx}/{len(pair_batches)} : pairs={len(pair_batch)}")

        # 把該段 pairs -> diff → seeds 映射
        diff2seeds = defaultdict(list)
        for diff, seed in pair_batch:
            diff2seeds[diff].append(seed)

        # 針對這段重新建立 tasks1
        tasks1 = []
        for ctrl in teacher_names:
            for diff, seed_list in diff2seeds.items():
                for i in range(0, len(seed_list), SEED_BATCH):
                    batch = seed_list[i : i + SEED_BATCH]
                    tasks1.append((ctrl, diff, batch,
                                   env_id, eval_n_stage1,
                                   reward_threshold, tmp_dir))

        print(f"    ↳ Stage-1 tasks this batch = {len(tasks1)}")
        chunks1 = max(1, len(tasks1) // (num_workers * 4))

        with Pool(processes=num_workers,
                  initializer=_init_worker,
                  initargs=(model_paths, device_str)) as pool:

            for ctrl, fit_res, paths in tqdm(
                    pool.imap_unordered(run_fitness_task, tasks1, chunksize=chunks1),
                    total=len(tasks1), desc="    Stage-1"):
                f = fitness[ctrl]
                f["total_reward"] += fit_res["total_reward"]
                f["episodes"]     += fit_res["episodes"]
                f["passed"]       += fit_res["passed"]
                paths_map[ctrl].extend(paths)

        # ----- 釋放 Pool / env / GPU 記憶體，再進下一段 -----

    # --------- Top-K 挑選 ----------
    fitness_out = {
        n: {"avg_reward": f["total_reward"]/max(1,f["episodes"]),
            "passed": f["passed"], "episodes": f["episodes"]}
        for n, f in fitness.items()
    }
    with open(os.path.join(save_dir, "fitness_stage1.json"), "w", encoding="utf-8") as fj:
        json.dump(fitness_out, fj, indent=2, ensure_ascii=False)

    top_names = sorted(fitness_out,
                       key=lambda n: fitness_out[n]["avg_reward"],
                       reverse=True)[:min(top_k, len(teacher_names))]
    print("\nTop-K controllers:", top_names[:5], "..." if len(top_names)>5 else "")
    fitness_top = {name: fitness_out[name] for name in top_names}

# 2. 將這個只包含 Top-K 成績的字典，存成新的 JSON 檔案
    top_k_filename = os.path.join(save_dir, "fitness_top.json")
    with open(top_k_filename, "w", encoding="utf-8") as fj:
        json.dump(fitness_top, fj, indent=2, ensure_ascii=False)

    print(f"Top-{len(top_names)} fitness scores saved to {top_k_filename}")
    # --------- Stage-2 (offline label) ----------
    label_tasks = [
        (p, top_names) for ctrl in top_names for p in paths_map[ctrl]
    ]
    chunks2 = max(1, len(label_tasks) // (num_workers * 4))

    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(model_paths, "cuda")) as pool:
        list(tqdm(pool.imap_unordered(offline_label_task, label_tasks, chunksize=chunks2),
                  total=len(label_tasks), desc="Stage-2 label"))

    # --------- 合併並存檔（與你原本相同） ----------
    S,M,LS,A,C,D,Z,V = [],[],[],[],[],[],[],[]
    for lp in tqdm([p.replace(".npz", ".label.npz") for p,_ in label_tasks], desc="Merging"):
        sp = lp.replace(".label.npz", ".npz")
        s  = np.load(sp); l = np.load(lp)
        S.append(s["states"].astype(np.float32))
        D.append(np.full((S[-1].shape[0],), float(s["diff"]), np.float32))
        Z.append(np.full((S[-1].shape[0],), int(np.int64(s["seed"])), np.int64))
        M.append(l["teacher_means"]); LS.append(l["teacher_logstds"])
        A.append(l["teacher_actions"]); C.append(l["confidences"])
        V.append(l["teacher_values"])   
    torch.save({
        "states": torch.from_numpy(np.concatenate(S,0)),
        "teacher_means":   torch.from_numpy(np.concatenate(M,0)),
        "teacher_logstds": torch.from_numpy(np.concatenate(LS,0)),
        "teacher_actions": torch.from_numpy(np.concatenate(A,0)),
        "confidences":     torch.from_numpy(np.concatenate(C,0)),
        "teacher_names": top_names,
        "difficulties": torch.from_numpy(np.concatenate(D,0)),
        "seeds":        torch.from_numpy(np.concatenate(Z,0)).long(),
        "teacher_values":  torch.from_numpy(np.concatenate(V,0)),
    }, os.path.join(save_dir, save_name))

    print(f"\n✅ Done. steps={sum(x.shape[0] for x in S)}, Top-K={len(top_names)}")
    return fitness_out, top_names
def _index_get(mod,key):
    if key.isdigit():#是否包含數字
        idx = int(key)
        if isinstance(mod,(nn.Sequential,nn.ModuleList,list,tuple)):
            return mod[idx]
        raise TypeError(f"{type(mod)}不適可索引容器:{key}")
    if isinstance(mod,nn.ModuleDict):
        if key in mod:return mod[key]
        raise KeyError(f"ModuelDict無鍵:{key}")
    if hasattr(mod,key):
        return getattr(mod,key)
    raise AttributeError(f"{type(mod)} 無屬性：{key}")
def _index_set(mod, key, new):
    """對 Sequential/ModuleList/ModuleDict 設定元素；否則用 setattr。"""
    if key.isdigit():
        idx = int(key)
        if isinstance(mod, (nn.Sequential, nn.ModuleList, list)):
            mod[idx] = new
            return
        raise TypeError(f"當前模組 {type(mod)} 不是可索引容器，卻用數字索引 {key}")
    if isinstance(mod, nn.ModuleDict):
        mod[key] = new
        return
    setattr(mod, key, new)
def resolve_parent_and_key(root, path: str):
    """回傳 (parent_module, last_key)。"""
    parts = path.split(".")
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for p in parts[:-1]:
        parent = _index_get(parent, p)
    return parent, parts[-1]
def get_module_by_path(root, path: str):
    mod = root
    for p in path.split("."):
        mod = _index_get(mod, p)
    return mod
def set_module_by_path(root, path: str, new_layer: nn.Module):
    """
    把路徑指定的那一層替換成 new_layer。
    例：set_module_by_path(model.policy, "mlp_extractor.policy_net.0", nn.Linear(24,64))
    """
    parent, last = resolve_parent_and_key(root, path)

    # 讓新層的 device/dtype 盡量和舊層一致（避免不小心留在 CPU）
    try:
        old = _index_get(parent, last)
        # 若舊層有參數，就用它的第一個參數的 device/dtype 對齊
        for p in old.parameters():
            new_layer = new_layer.to(device=p.device, dtype=p.dtype)
            break
    except Exception:
        pass  # 對齊失敗也沒關係，不阻塞替換

    _index_set(parent, last, new_layer)
def is_param_layer(m:nn.Module) -> bool:
    return isinstance(m,(nn.Linear,nn.Conv2d))
def param_layer_indices(seq: nn.Sequential):
    return [i for i,m in enumerate(seq) if is_param_layer(m)]
@dataclass
class Gene:
    seq_path: str
    idx: int
    kind: str
    shape: tuple
def collect_genes(policy) -> List[Gene]:
    genes: List[Gene] = []
    seq_paths = [
        "mlp_extractor.policy_net",   # Actor 隱藏
        "mlp_extractor.value_net",    # Critic 隱藏
    ]
    # 頭部（非 Sequential）：當作單獨基因（不插適應層）
    head_paths = ["action_net", "value_net"]
    for sp in seq_paths:
        seq = get_module_by_path(policy, sp) 
        for i in param_layer_indices(seq):
            m =seq[i]
            kind = "linear" if isinstance(m,nn.Linear)else"conv2d"
            genes.append(Gene(seq_path=sp,idx=i,kind=kind,shape=tuple(m.weight.shape)))
        for hp in head_paths:
            m = getattr(policy, hp)
            kind = "linear" if isinstance(m, nn.Linear) else "conv2d"
            genes.append(Gene(seq_path=hp, idx=-1, kind=kind, shape=tuple(m.weight.shape)))
    return genes
class ScaleShift(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
    def forward(self,x):return self.gamma*x+self.beta
class BoundaryAdapter(nn.Module):
    def __init__(self,core:nn.Module,kind:str,in_dim: int,out_dim: int):
        super().__init__()
        self.core ,self.kind,self.in_dim,self.out_dim=core,kind,in_dim,out_dim
        self.is_adapter=True
    def forward(self,x): return self.core(x)
def make__adapter(prev: nn.Module,nxt:nn.Module)->nn.Module:
    if isinstance(prev,nn.Linear) and isinstance(nxt,nn.Linear):
        in_d,out_d =prev.out_features,nxt.in_features
        if in_d==out_d:
            return BoundaryAdapter(ScaleShift((out_d,)),"scaleshift",in_d,out_d)
        core = nn.Linear (in_d,out_d,bias=True)
        nn.init.orthogonal_(core.weight);nn.init.zeros_(core.bias)
        return BoundaryAdapter(core,"linear",in_d,out_d)
    if isinstance(prev, nn.Conv2d) and isinstance(nxt, nn.Conv2d):
        in_c, out_c = prev.out_channels, nxt.in_channels
        if in_c == out_c:
            return BoundaryAdapter(ScaleShift((out_c,1,1)), "scaleshift", in_c, out_c)
        core = nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(core.weight, a=math.sqrt(5)); nn.init.zeros_(core.bias)
        return BoundaryAdapter(core, "conv1x1", in_c, out_c)
    raise NotImplementedError("如需 Conv↔Linear（含 Flatten）可再加 probe。")

@torch.no_grad()
def copy_params(dst:nn.Module,src:nn.Module):
    if hasattr(dst,"weight") and hasattr(src,"weight") and dst.weight.shape ==  src.weight.shape:
        dst.weight.data.copy_(src.weight.data) 
    if getattr(dst,"bias",None) is not None and getattr(src,"bias",None) is not None:
        if dst.bias.shape == src.bias.shape:
            dst.bias.data.copy_(src.bias.data)
def insert_between(seq: nn.Sequential,idx_prev:int,adapter:nn.Module) -> nn.Sequential:
    layers=[]
    for i ,m in enumerate(seq):
        layers.append(m)
        if i == idx_prev:
            layers.append(adapter)
    return nn.Sequential(*layers)
@torch.no_grad()
def crossover_one_boundary(
    child_policy: nn.Module,
    father_policy: nn.Module,
    mother_policy: nn.Module,
    p_from_mother: float = 0.5,     # 保留介面相容性（本版不使用）
    device: str = "cuda",
    forbidden: Optional[Dict[str, Set[int]]] = None,
    debug_target: Optional[Tuple[str, int]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    每輪只交配一個「單位」：
      - 單位可以是：MLP extractor 的單一參數層，或整個 head(action_net/value_net)
      - 以父為基底，只覆蓋母親到該單位；其餘保持父
      - 若交配的是 MLP 層：在第一個 M/F 邊界插入 1 個 adapter（同維度→ScaleShift）
      - forbidden: {"mlp_extractor.policy_net": {idx1, ...}, "action_net": {-1}, ...}
    回傳 info: switches, adapter, touched
    """
    if forbidden is None:
        forbidden = {}

    BACKBONES = ("mlp_extractor.policy_net", "mlp_extractor.value_net")
    HEADS     = ("action_net", "value_net")  # 注意：與 backbone 的 value_net 不同路徑

    switches: Dict[str, Dict[int, str]] = {}
    touched:  Dict[str, Set[int]] = {}

    # ------- 蒐集候選：MLP 參數層 + 兩個 head（排除 forbidden） -------
    candidates: List[Tuple[str, Optional[int]]] = []

    # MLP 參數層
    for sp in BACKBONES:
        seq = get_module_by_path(child_policy, sp)
        for idx in param_layer_indices(seq):
            if sp in forbidden and idx in forbidden[sp]:
                continue
            candidates.append((sp, idx))

    # head：用 -1 當索引
    for sp in HEADS:
        if sp in forbidden and (-1 in forbidden[sp]):
            continue
        candidates.append((sp, -1))

    # 若沒有可用候選，直接返回
    if not candidates:
        return child_policy, dict(switches=switches, adapter=None, touched=touched)

    # ------- 隨機選「一個」單位交配 -------
    if debug_target and debug_target in candidates:
        # 如果提供了有效的 debug_target，就使用它
        target_sp, target_idx = debug_target
        print(f"[DEBUG PICK] unit={target_sp} idx={target_idx}")
    else:
        # 否則，恢復隨機選擇
        target_sp, target_idx = random.choice(candidates)
        print(f"[RANDOM PICK] unit={target_sp} idx={target_idx}")

    # 如果選到 head：整塊由母親覆蓋，其它不動；不插 adapter
    if target_sp in HEADS:
        dst = getattr(child_policy, target_sp)
        src = getattr(mother_policy, target_sp)
        copy_params(dst, src)
        touched.setdefault(target_sp, set()).add(-1)
        # head 不參與 switches/adapter（不在 backbone 上）
        return child_policy, dict(switches=switches, adapter=None, touched=touched)

    # ------- 若選到 MLP 層：只該層覆蓋母親，其餘標記為父 -------
    # 先把兩條 backbone 的參數層都標記好（選中層 M，其他 F）
    for sp in BACKBONES:
        seq_c   = get_module_by_path(child_policy, sp)
        seq_mom = get_module_by_path(mother_policy, sp)
        idxs    = param_layer_indices(seq_c)

        for idx in idxs:
            if sp == target_sp and idx == target_idx:
                # 覆蓋母親該層
                copy_params(seq_c[idx], seq_mom[idx])
                switches.setdefault(sp, {})[idx] = "M"
                touched.setdefault(sp, set()).add(idx)
            else:
                switches.setdefault(sp, {})[idx] = "F"  # 保持父親（child 已是父 clone，無需 copy）

    # ------- 在第一個 M/F 邊界插入 1 個 adapter（同維度→ScaleShift） -------
    adapters_info = None
    def _make_scaleshift_adapter(prev_layer: nn.Module, next_layer: nn.Module) -> nn.Module:
        feat = getattr(next_layer, "in_features", None)
        if feat is None:
            feat = getattr(prev_layer, "out_features", None)
        if feat is None:
            raise TypeError(f"無法推斷 ScaleShift 維度，prev={type(prev_layer)}, next={type(next_layer)}")
        ss = ScaleShift((feat,))
        return BoundaryAdapter(ss, "scaleshift", feat, feat)

    for sp in BACKBONES:
        seq  = get_module_by_path(child_policy, sp)
        idxs = param_layer_indices(seq)
        marks = switches.get(sp, {})

        # 找第一個相鄰參數層的 M/F 切換
        for j in range(len(idxs) - 1):
            i_prev, i_next = idxs[j], idxs[j + 1]
            left_src, right_src = marks.get(i_prev), marks.get(i_next)
            if left_src is None or right_src is None:
                continue
            if left_src != right_src:
                insert_pos = i_next - 1
                adp = make__adapter(seq[i_prev], seq[i_next]).to(device)
                # 保險：若工廠回傳 Identity，換成 ScaleShift（通常你的工廠已回 scaleshift）
                if isinstance(adp.core, nn.Identity):
                    adp = _make_scaleshift_adapter(seq[i_prev], seq[i_next]).to(device)
                new_seq = insert_between(seq, insert_pos, adp)
                set_module_by_path(child_policy, sp, new_seq)
                adapters_info = dict(
                    seq_path=sp,
                    between=(i_prev, i_next),
                    adapter=type(adp.core).__name__,
                )
                break
        if adapters_info:
            break  # 本輪只插 1 個
    return child_policy, dict(switches=switches, adapter=adapters_info, touched=touched)
def freeze_except_adapters(model:nn.Module):
    for p in model.parameters(): p.requires_grad =False
    for m in model.modules():
        if getattr(m,"is_adapter",False):
            for p in m.parameters():p.requires_grad=True
def unfreeze_all(model:nn.Module):
    for p in model.parameters(): p.requires_grad =True
def train_adapters_offline(
    ppo_model: PPO,
    pt_path: str,
    adapter_info: dict, # <--【新增】接收來自 crossover 的 info['adapter']
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 4096,
    lambda_std: float = 0.05,
    device: str = "cuda",
    temperature =0.5 ,
):
    blob = torch.load(pt_path, map_location="cuda")
    states = blob["states"].float()  # [N, obs_dim]
    means  = blob["teacher_means"].float()  # [N,Tchr,act_dim]
    logstds_t = blob["teacher_logstds"].float()
    conf   = blob["confidences"].float()    # [N,Tchr]
    values_t = blob["teacher_values"].float()
    # 加權平均老師 mean 作目標
    # --- 【新的目標準備邏輯：只聽最自信的老師】 ---

# 1. 對於每一個狀態，找到信賴度最高的老師的索引
#    conf 的形狀是 [N, Tchr]，argmax(dim=1) 會返回每個樣本中最大值的索引
    best_teacher_indices = torch.argmax(conf, dim=1)  # 結果形狀為 [N]

    # 2. 準備一個索引器，用於從原始數據中高效地提取數據
    N = states.shape[0]
    # 這是 PyTorch/NumPy 中非常高效的進階索引 (Advanced Indexing)
    idx_gather = torch.arange(N, device=device)

    # 3. 使用這個索引器，從原始數據中只挑選出「最佳老師」的預測作為目標
    mu_tgt = means[idx_gather, best_teacher_indices, :]
    logstd_tgt = logstds_t[idx_gather, best_teacher_indices, :]
    value_tgt = values_t[idx_gather, best_teacher_indices]

# --- 後續程式碼不變 ---
    print("\n" + "="*20 + " Shape Debugging " + "="*20)
    print(f"Loaded 'states' shape:           {states.shape}")
    print(f"Calculated 'mu_tgt' shape:       {mu_tgt.shape}")
    print(f"Calculated 'logstd_tgt' shape:    {logstd_tgt.shape}")
    print(f"Calculated 'value_tgt' shape:      {value_tgt.shape}  <--- 請重點關注這一行！")
    print("="*57 + "\n")
    train_target ="policy" if "policy" in adapter_info['seq_path'] else "critic"
    print(f"->Targeting new adapter in'{adapter_info['seq_path']}'. Training mode: {train_target}")
    if train_target == "policy":
        ds =torch.utils.data.TensorDataset(states,mu_tgt,logstd_tgt)
    else:
        ds =torch.utils.data.TensorDataset(states,value_tgt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    policy = ppo_model.policy.to(device)
    freeze_except_adapters(policy)
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f"DEBUG: Found {len(params)} trainable parameters.")
    print(f"DEBUG: Trainable parameter shapes: {[p.shape for p in params]}")
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    def get_gaussian(obs_batch):
        f = policy.extract_features(obs_batch)
        lat_pi, lat_vf = policy.mlp_extractor(f) if policy.share_features_extractor \
                   else policy.mlp_extractor(f, None)
        mu = policy.action_net(lat_pi)
        log_std = policy.log_std.expand_as(mu) if hasattr(policy, "log_std") else torch.zeros_like(mu)
        value = policy.value_net(lat_vf)
        return mu, log_std,value

    policy.train()
    with torch.enable_grad():
        for ep in range(epochs):
            run_l, n = 0.0, 0
            for data_batch in dl:
                data_batch=[t.to(device)for t in data_batch]
                obs =data_batch[0]
                opt.zero_grad(set_to_none=True) # 清除上一輪的梯度
                mu_s,logstds_s,values_s =get_gaussian(obs)
                if train_target=="policy":
                    # mu_t,logstds_t=data_batch[1],data_batch[2]
                    # var_t = torch.exp(2 * logstds_t)
                    # var_s = torch.exp(2 * logstds_s)
                    # if temperature != 1.0:
                        
                    # # 確保溫度是一個張量以便進行廣播運算
                    #     temp_tensor = torch.tensor(temperature, device=device)
                    #     # 調整教師的 logstd 和 variance
                    #     var_t = var_t * (temp_tensor**2)
                    #     logstds_t = logstds_t + torch.log(temp_tensor)
                    # kl_div = (logstds_s - logstds_t) + (var_t + (mu_t - mu_s).pow(2)) / (2 * var_s) - 0.5
                    # loss = kl_div.sum(dim=-1).mean()
                    mu_t, logstds_t = data_batch[1], data_batch[2]
                    loss_mu  = torch.nn.functional.mse_loss(mu_s, mu_t)
                    loss_std = torch.nn.functional.mse_loss(logstds_s, logstds_t)

                    loss = loss_mu 
                else:
                    values_t= data_batch[1]
                    loss=((values_s.squeeze()-values_t)**2).mean()
                loss.backward()
                # if len(params) > 0:
                #     total_norm = 0
                #     for p in params:
                #         if p.grad is not None:
                #             param_norm = p.grad.data.norm(2)
                #             total_norm += param_norm.item() ** 2
                #     total_norm = total_norm ** 0.5
                #     # 在 print loss 前面或後面加上這一行
                #     print(f"  -> Grad Norm: {total_norm:.6f}", end="")
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                run_l += loss.item() * obs.size(0); n += obs.size(0)
            loss_name = "KL_Loss" if train_target == "policy" else "MSE_Loss"
            print(f"  Adapter epoch {ep+1}/{epochs} \t{loss_name}={run_l/n:.6f}")
    policy.eval()
def train_network_offline(
    ppo_model: PPO,
    pt_path: str,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 4096,
    vf_coef: float = 0.5,
    temperature: float = 2.0,  # 軟目標通常需要稍高的溫度
    alpha: float = 0.9,      # 新增 alpha 參數，控制軟硬目標的權重
    device: str = "cuda",
    
):
    blob = torch.load(pt_path, map_location="cuda")
    states = blob["states"].float()  # [N, obs_dim]
    means  = blob["teacher_means"].float()  # [N,Tchr,act_dim]
    logstds_t = blob["teacher_logstds"].float()
    conf   = blob["confidences"].float()    # [N,Tchr]
    values_t = blob["teacher_values"].float()
    # 加權平均老師 mean 作目標
    # --- 【新的目標準備邏輯：只聽最自信的老師】 ---

# 1. 對於每一個狀態，找到信賴度最高的老師的索引
#    conf 的形狀是 [N, Tchr]，argmax(dim=1) 會返回每個樣本中最大值的索引
    best_teacher_indices = torch.argmax(conf, dim=1)  # 結果形狀為 [N]

    # 2. 準備一個索引器，用於從原始數據中高效地提取數據
    N = states.shape[0]
    # 這是 PyTorch/NumPy 中非常高效的進階索引 (Advanced Indexing)
    idx_gather = torch.arange(N, device="cpu")

    # 3. 使用這個索引器，從原始數據中只挑選出「最佳老師」的預測作為目標
    mu_tgt = means[idx_gather, best_teacher_indices, :]
    logstd_tgt = logstds_t[idx_gather, best_teacher_indices, :]
    value_tgt = values_t[idx_gather, best_teacher_indices]

# --- 後續程式碼不變 ---
    print("\n" + "="*20 + " Shape Debugging " + "="*20)
    print(f"Loaded 'states' shape:           {states.shape}")
    print(f"Calculated 'mu_tgt' shape:       {mu_tgt.shape}")
    print(f"Calculated 'logstd_tgt' shape:    {logstd_tgt.shape}")
    print(f"Calculated 'value_tgt' shape:      {value_tgt.shape}  <--- 請重點關注這一行！")
    print("="*57 + "\n")
    ds =torch.utils.data.TensorDataset(states, mu_tgt, logstd_tgt,value_tgt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    policy = ppo_model.policy.to(device)
    for param in policy.parameters():
        param.requires_grad = True
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f"DEBUG: Found {len(params)} trainable parameters.")
    print(f"DEBUG: Trainable parameter shapes: {[p.shape for p in params]}")
    #opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    def get_gaussian(obs_batch):
        f = policy.extract_features(obs_batch)
        lat_pi, lat_vf = policy.mlp_extractor(f) if policy.share_features_extractor \
                   else policy.mlp_extractor(f, None)
        mu = policy.action_net(lat_pi)
        log_std = policy.log_std.expand_as(mu) if hasattr(policy, "log_std") else torch.zeros_like(mu)
        value = policy.value_net(lat_vf)
        return mu, log_std,value

    policy.train()
    with torch.enable_grad():
        for ep in range(epochs):
            run_pi_l, run_vf_l,n = 0.0, 0.0,0
            for data_batch in dl:
                data_batch=[t.to(device)for t in data_batch]
                obs, mu_t, logstds_t, values_t = data_batch # 解包所有數據
                optimizer.zero_grad(set_to_none=True) # 清除上一輪的梯度
                mu_s,logstds_s,values_s =get_gaussian(obs)
                
                 # 1. 計算學生和老師的 variance (方差)
                var_s = torch.exp(2 * logstds_s)
                logstds_t_soft = logstds_t
                # 2. 根據溫度調整老師的目標分佈
                if temperature > 1.0:
                    temp_tensor = torch.tensor(temperature, device=device)
                    # 溫度越高，老師的 log_std 越大，分佈越寬
                    logstds_t_soft = logstds_t + torch.log(temp_tensor)
                
                var_t = torch.exp(2 * logstds_t_soft)

                # 3. 計算 KL 散度 KL(P_teacher || P_student)
                # 這是衡量兩個高斯分佈差異的標準公式
                kl_div = (logstds_s - logstds_t_soft) + (var_t + (mu_t - mu_s).pow(2)) / (2 * var_s) - 0.5
                
                # Policy 損失：對 KL 散度在動作維度上求和，然後在 batch 維度上取平均
                loss_soft = kl_div.sum(dim=-1).mean()
                loss_hard=torch.nn.functional.mse_loss(mu_s,mu_t)
                loss_pi=alpha*loss_soft+(1-alpha)*loss_hard
                # Value 損失 (保持不變)
                loss_vf = torch.nn.functional.mse_loss(values_s.squeeze(), values_t)
                
                # 總損失
                loss = loss_pi + vf_coef * loss_vf
                # -----------------------------------------------

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                
                run_pi_l += loss_pi.item() * obs.size(0)
                run_vf_l += loss_vf.item() * obs.size(0)
                n += obs.size(0)
            avg_pi_loss = run_pi_l / n
            avg_vf_loss = run_vf_l / n
            # 在損失名稱上做個區分
            print(f"  Offline Epoch {ep+1}/{epochs} \t Policy_KL_Loss={avg_pi_loss:.6f} \t Value_MSE_Loss={avg_vf_loss:.6f}")
            
    policy.eval()
def _rebuild_policy_optimizer(policy: nn.Module):
    """
    跨 SB3 版本安全地重建 policy 的 optimizer。
    - 先取目前 requires_grad=True 的參數
    - 优先用 policy.setup_optimizer()（若存在）
    - 否則手動用 optimizer_class/optimizer_kwargs 建
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found when rebuilding optimizer.")

    # 一些 SB3 版本有 setup_optimizer()
    if hasattr(policy, "setup_optimizer") and callable(policy.setup_optimizer):
        # 注意：某些版本的 setup_optimizer 會對 self.parameters() 建 optimizer，
        # 會尊重 requires_grad 狀態；你已在外面設好 requires_grad 了。
        policy.setup_optimizer()
    else:
        # 後備方案：手動建立
        opt_cls = getattr(policy, "optimizer_class", torch.optim.Adam)
        opt_kwargs = getattr(policy, "optimizer_kwargs", {})
        policy.optimizer = opt_cls(trainable, **opt_kwargs)
def progressive_evolve(
    dad: PPO, mom: PPO, env, distill_pt,
    rounds: int = 1, finetune_steps: int = 150_00,debug_plan: Optional[Dict[int, Tuple[str, int]]] = None,
):
    child_policy = copy.deepcopy(dad.policy).to("cuda")  # 以父為骨架
    forbidden: Dict[str,Set[int]]={}
    for r in range(1, rounds+1):
        print(f"\n================  Round {r}  ================")
        round_debug_target = debug_plan.get(r) if debug_plan else None
        child_policy, info = crossover_one_boundary(
            child_policy, dad.policy, mom.policy,
            p_from_mother=0.5, device="cuda",forbidden=forbidden,debug_target=round_debug_target)
        print("插入 info:", info["adapter"])
        touched = info.get("touched", {})
        for sp, idx_set in touched.items():
            forbidden.setdefault(sp, set()).update(idx_set)
        # 用 child_policy 建一隻暫時的 PPO（共用超參）
        child = PPO(
            policy=dad.policy.__class__,
            env=env,
            verbose=0,
            learning_rate=3e-5,
            n_steps=dad.n_steps,
            batch_size=dad.batch_size,
            n_epochs=dad.n_epochs,
            gamma=dad.gamma,
            gae_lambda=dad.gae_lambda,
            clip_range=dad.clip_range,
            ent_coef=dad.ent_coef,
        )
        child.policy = child_policy
        child.policy.to(child.device)
        _rebuild_policy_optimizer(child.policy)  # 讓 optimizer 知道哪些要訓練
        if info.get("adapter") is None:
            print("[offline] no new adapter this round -> skip offline distill")
        else:
            train_adapters_offline(
                child, distill_pt,adapter_info=info["adapter"],
                epochs=2, lr=1e-3, batch_size=4096,
                lambda_std=0.05, device="cuda", temperature=2.0  # 或 "cuda"
    )
        # 解凍全網 finetune
        unfreeze_all(child.policy)
        _rebuild_policy_optimizer(child.policy)
        child.set_env(env)
        child.learn(total_timesteps=finetune_steps, progress_bar=True)

        # 更新父引用 → 下一輪以最新子網再交配
        dad.policy = child.policy
        child_policy = child.policy

    return dad
def _unwrap_policy(x):
    return getattr(x, "policy", x)


def _safe_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu")

def print_and_dump_all_params(
    model: Union[nn.Module, object],
    who: str,
    out_dir: str = "./weight_dump",
    to_files: bool = True,
    print_full: bool = True,
    float_fmt: str = "{:.7g}",
):
    """
    完整列印 + 另存所有參數（weights/bias/ln 等）。
    - model: 可是 PPO 或 policy
    - who:   "dad" / "mom" / "child"（會用在檔名）
    - to_files: 另存到 out_dir/who/ 下（.pt + 各層 .txt）
    - print_full: True = 直接把整個 tensor 印到主控台
    """
    pol = _unwrap_policy(model)
    os.makedirs(os.path.join(out_dir, who), exist_ok=True)

    # 讓列印更完整（避免換行截斷）
    if print_full:
        torch.set_printoptions(sci_mode=False, linewidth=10_000, threshold=10_000_000)

    print(f"\n================= [{who}] 參數完整列印開始 =================")
    meta = {}

    for name, p in pol.named_parameters():
        t = _safe_cpu(p)
        shp = tuple(t.shape)
        meta[name] = {"shape": shp, "numel": t.numel()}

        header = f"\n--- {who} :: {name}  shape={shp}  numel={t.numel()} ---"
        print(header)
        if print_full:
            print(t)  # 直接完整印出
        else:
            # 只印摘要，避免刷屏
            print(f"min={t.min().item():.6g}  max={t.max().item():.6g}  mean={t.mean().item():.6g}  std={t.std().item():.6g}")

        if to_files:
            # 逐層文字檔（.txt）
            txt_path = os.path.join(out_dir, who, f"{name.replace('.', '_')}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(header + "\n")
                if print_full:
                    f.write(str(t) + "\n")
                else:
                    f.write(f"[summary] min={t.min().item():.9g} max={t.max().item():.9g} mean={t.mean().item():.9g} std={t.std().item():.9g}\n")

    # 存一份單檔 .pt（整包 state_dict）
    if to_files:
        state_pt = {k: _safe_cpu(v) for k, v in pol.state_dict().items()}
        torch.save(state_pt, os.path.join(out_dir, f"{who}_state_dict.pt"))
        with open(os.path.join(out_dir, f"{who}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n================= [{who}] 參數列印完成 =================\n"f"{'(已輸出至 ' + os.path.abspath(out_dir) + ')' if to_files else ''}")
@torch.no_grad()
def create_dual_channel_policy(dad_policy:nn.Module,mom_policy:nn.Module)->nn.Module:
    print("正在建構網路中.....")
    child_policy=copy.deepcopy(dad_policy)
    dad_modules=dict(dad_policy.named_modules())
    mom_modules=dict(mom_policy.named_modules())
    for name,child_module in child_policy.named_modules():
        if isinstance(child_module,nn.Linear):
            dad_layer= dad_modules.get(name)
            mom_layer =mom_modules.get(name)
            if dad_layer is None or mom_layer is None:
                continue
            dad_w=dad_layer.weight.data
            mom_w=mom_layer.weight.data
            if dad_w.shape[0] % 2 != 0:
                print(f"  - 警告：層 '{name}' 的輸出維度 ({dad_w.shape[0]}) 為奇數，已跳過拼接。")
                continue
            out_half =dad_w.shape[0]//2
            assert dad_w.shape[0] % 2 == 0, f"Layer {name} out_features not divisible by 2"
            child_module.weight.data[:out_half,:]=dad_w[:out_half,:]
            child_module.weight.data[out_half:,:]=mom_w[out_half:,:]
            if child_module.bias is not None:
                dad_b=dad_layer.bias.data
                mom_b=mom_layer.bias.data
                child_module.bias.data[:out_half]=dad_b[:out_half]
                child_module.bias.data[out_half:]=mom_b[out_half:]

    print("網路建構完成")
    # verify_half_concat_mom_lower(child_policy,dad.policy,mom.policy)
    # print_and_dump_all_params(dad,   "dad",   to_files=True,  print_full=True)  # 爸爸
    # print_and_dump_all_params(mom,   "mom",   to_files=True,  print_full=True)  # 媽媽
    # print_and_dump_all_params(child_policy, "child", to_files=True,  print_full=True)
    return child_policy

@torch.no_grad()
def verify_half_concat_mom_lower(child: nn.Module, dad: nn.Module, mom: nn.Module, atol=1e-6) -> bool:
    """
    驗證「上半=爸爸上半，下半=媽媽下半」的拼接設計。
    """
    child = _unwrap_policy(child)
    dad = _unwrap_policy(dad)
    mom = _unwrap_policy(mom)
    ok = True
    dad_modules = dict(dad.named_modules())
    mom_modules = dict(mom.named_modules())

    for name, m in child.named_modules():
        if not isinstance(m, nn.Linear):
            continue

        d = dad_modules.get(name)
        mo = mom_modules.get(name)
        if d is None or mo is None:
            print(f"[SKIP] {name}: 找不到父母對應層")
            continue

        Wc, Wd, Wm = m.weight.data, d.weight.data, mo.weight.data
        if Wc.shape != Wd.shape or Wc.shape != Wm.shape or Wc.shape[0] % 2 != 0:
            print(f"[SKIP] {name}: 形狀不符 -> child:{Wc.shape}, dad:{Wd.shape}, mom:{Wm.shape}")
            continue

        oh = Wc.shape[0] // 2

        # 🧩 確認上半、下半分別來自父母正確區段
        diff_dad = (Wc[:oh, :] - Wd[:oh, :]).abs().max().item()     # 上半 = 爸上半
        diff_mom = (Wc[oh:, :] - Wm[oh:, :]).abs().max().item()     # 下半 = 媽下半
        cond_w = diff_dad <= atol and diff_mom <= atol

        # 偏置比對
        cond_b = True
        diff_b_dad = diff_b_mom = None
        if m.bias is not None:
            bd, bm, bc = d.bias.data, mo.bias.data, m.bias.data
            diff_b_dad = (bc[:oh] - bd[:oh]).abs().max().item()
            diff_b_mom = (bc[oh:] - bm[oh:]).abs().max().item()
            cond_b = diff_b_dad <= atol and diff_b_mom <= atol

        if cond_w and cond_b:
            print(f"[OK] {name:40s}  diff_dad={diff_dad:.3e}  diff_mom={diff_mom:.3e}")
        else:
            print(f"[X] {name:40s}  diff_dad={diff_dad:.3e}  diff_mom={diff_mom:.3e}")
            ok = False

    print("\n✅ 全部符合『上半=爸上半，下半=媽下半』" if ok else "\n⚠️ 有層未通過比對")
    return ok
@torch.no_grad()
def zero_initialize_crosstalk(policy: nn.Module):
    """
    將雙子網路中所有線性層的交叉通道權重歸零。
    """
    print("正在執行交叉通道歸零...")
    for name, module in policy.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            in_half = W.shape[1] // 2
            out_half = W.shape[0] // 2
            
            # 確保維度有效
            if in_half == 0 or out_half == 0: continue

            # # 左下角 (母 -> 父) 區塊
            # W[:out_half, in_half:].zero_()
            # # 右上角 (父 -> 母) 區塊
            # W[out_half:, :in_half].zero_()
            #--- 核心修改：從 .zero_() 改為 nn.init.normal_ ---
            ##選中右上角 (母 -> 父) 區塊
            cross_mf_block = W[:out_half, in_half:]
            nn.init.normal_(cross_mf_block, mean=0.0, std=0.001)

            # 選中左下角 (父 -> 母) 區塊
            cross_fm_block = W[out_half:, :in_half]
            nn.init.normal_(cross_fm_block, mean=0.0, std=0.001)
    print("歸零完成。")
    # print_and_dump_all_params(dad,   "dad",   to_files=True,  print_full=True)  # 爸爸
    # print_and_dump_all_params(mom,   "mom",   to_files=True,  print_full=True)  # 媽媽
    # print_and_dump_all_params(child_policy, "child", to_files=True,  print_full=True)
@torch.no_grad()
def verify_zero_initialize_crosstalk(policy_before: nn.Module, policy_after: nn.Module, atol=1e-8):
    """
    驗證 zero_initialize_crosstalk() 是否僅修改 cross-block。
    檢查：
      - 父->父 (左上) 與 母->母 (右下) 權重區塊保持一致
      - cross-block (右上、左下) 被改動 (差異 > atol)
    """
    ok = True
    before_modules = dict(policy_before.named_modules())
    after_modules  = dict(policy_after.named_modules())

    for name, m_after in after_modules.items():
        if not isinstance(m_after, nn.Linear):
            continue
        m_before = before_modules.get(name)
        if m_before is None:
            continue

        Wb = m_before.weight.data.clone()
        Wa = m_after.weight.data.clone()

        if Wb.shape != Wa.shape or Wb.ndim != 2:
            continue

        out_half, in_half = Wa.shape[0] // 2, Wa.shape[1] // 2
        if out_half == 0 or in_half == 0:
            continue

        # 區塊定義
        dad_dad_block = Wa[:out_half, :in_half]   # 父->父
        mom_mom_block = Wa[out_half:, in_half:]   # 母->母
        cross_mf_block = Wa[:out_half, in_half:]  # 母->父
        cross_fm_block = Wa[out_half:, :in_half]  # 父->母

        # 檢查哪些區塊被改動
        diff_dad = (dad_dad_block - Wb[:out_half, :in_half]).abs().max().item()
        diff_mom = (mom_mom_block - Wb[out_half:, in_half:]).abs().max().item()
        diff_cross1 = (cross_mf_block - Wb[:out_half, in_half:]).abs().max().item()
        diff_cross2 = (cross_fm_block - Wb[out_half:, :in_half]).abs().max().item()

        pass_pure = diff_dad <= atol and diff_mom <= atol
        pass_cross = (diff_cross1 > atol*10) or (diff_cross2 > atol*10)

        if pass_pure and pass_cross:
            print(f"[OK] {name:40s} 父母區未改動, cross區已更新 ✅")
        else:
            ok = False
            print(f"[X] {name:40s} diff_dad={diff_dad:.2e}, diff_mom={diff_mom:.2e}, "
                  f"diff_cross1={diff_cross1:.2e}, diff_cross2={diff_cross2:.2e}")

    print("\n✅ 通過全部層驗證" if ok else "\n⚠️ 有層不符合預期")
    return ok
def create_refined_differential_optimizer(
    policy: nn.Module, 
    lr_pure: float, 
    cross_lr_scale_factor: float = 0.01 # 將交叉學習率改為一個縮放因子
) -> torch.optim.Adam:
    """
    創建一個差異化學習率的優化器（精細版）。
    使用梯度掛鉤 (gradient hook) 來精準地縮放交叉通道的梯度。
    """
    print(f"創建精細版差異化優化器，基礎學習率: {lr_pure}, 交叉通道梯度縮放因子: {cross_lr_scale_factor}")
    
    # 步驟 1: 為所有參數設定一個統一的、較高的學習率
    # 我們不再需要手動分離參數組
    all_params = policy.parameters()
    optimizer = torch.optim.Adam(all_params, lr=lr_pure)

    # 步驟 2: 遍歷所有線性層的權重，並為它們註冊一個特製的掛鉤
    for name, module in policy.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight
            
            # 獲取維度資訊
            in_half = W.shape[1] // 2
            out_half = W.shape[0] // 2

            if in_half == 0 or out_half == 0:
                continue
            
            # --- 這是 Python 的一個技巧，稱為「閉包 (Closure)」---
            # 我們定義一個函式，它會「記住」當前層的維度和縮放因子
            # 然後返回我們真正需要的掛鉤函式
            def create_hook(oh=out_half, ih=in_half, sf=cross_lr_scale_factor):
                def hook(grad):
                    # 這是掛鉤的核心：修改梯度
                    with torch.no_grad():
                        # # 右上角 (母 -> 父) 的梯度，乘以縮放因子
                        # grad[:oh, ih:].mul_(sf)
                        # # 左下角 (父 -> 母) 的梯度，乘以縮放因子
                        # grad[oh:, :ih].mul_(sf)
                        grad[:oh, :ih].mul_(sf)
                        # 左下角 (父 -> 母) 的梯度，乘以縮放因子
                        grad[oh:, ih:].mul_(sf)
                        # grad[:oh, :ih].zero_()
                        # # 左下角 (父 -> 母) 的梯度，乘以縮放因子
                        # grad[oh:, ih:].zero_()
                    return grad
                return hook
            
            # 為當前權重張量 W 註冊這個新鮮創建的掛鉤
            W.register_hook(create_hook())

    print("所有權重矩陣的梯度掛鉤均已註冊。")
    return optimizer
def freeze_pure_channels(
    policy: nn.Module,
    *,
    bias_mode: str = "freeze",     # "freeze" | "train"
    allow_train_if_unsplit: bool = False  # in_half==0 or out_half==0 時是否整層可訓練
) -> Tuple[List[torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    凍結純淨通道（左上=父->父、右下=母->母），只允許『交叉通道』（右上、左下）學習。
    bias_mode:
      - "freeze": bias 也凍結（掛 hook 將梯度清零、且不加入 optimizer）
      - "train" : bias 允許學習（加入 optimizer）
    allow_train_if_unsplit:
      - True  : 對無法二分的層(oh==0 或 ih==0)整層可訓練
      - False : 這類層整層凍結
    回傳: (trainable_params, hook_handles)
    """
    print("--- 正在凍結純淨通道 (父->父, 母->母) ---")

    # 先關閉所有參數的 requires_grad
    for p in policy.parameters():
        p.requires_grad = False

    trainable_params: List[torch.Tensor] = []
    hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight
        oh = W.shape[0] // 2
        ih = W.shape[1] // 2

        # 無法二分的層
        if oh == 0 or ih == 0:
            if allow_train_if_unsplit:
                # 整層可訓練
                W.requires_grad = True
                trainable_params.append(W)
                if module.bias is not None and bias_mode == "train":
                    module.bias.requires_grad = True
                    trainable_params.append(module.bias)
            # else: 全凍結（不做任何事）
            continue

        # 權重：整張 weight 允許計算梯度，但用 hook 清掉純淨區塊梯度
        W.requires_grad = True
        def make_weight_hook(oh=oh, ih=ih):
            def hook(grad):
                # 清掉純淨塊：左上(父->父)、右下(母->母)
                grad[:oh, :ih] = 0.0
                grad[oh:, ih:] = 0.0
                return grad
            return hook
        h = W.register_hook(make_weight_hook())
        hook_handles.append(h)
        trainable_params.append(W)  # 交叉塊會有梯度 → 交給 optimizer

        # bias：依 bias_mode 決定是否訓練
        if module.bias is not None:
            B = module.bias
            if bias_mode == "freeze":
                # 凍結：掛 hook 把整個 bias 梯度清零（或只清純淨上/下半）
                B.requires_grad = True  # 需要 True 才會走 hook
                def make_bias_hook(oh=oh):
                    def hook(grad):
                        grad.zero_()         # 完全不更新 bias
                        # 若只想凍結純淨、放開交叉對應的 bias（通常不需要），可改成：
                        # grad[:oh] = 0.0
                        # grad[oh:] = 0.0
                        return grad
                    return hook
                hook_handles.append(B.register_hook(make_bias_hook()))
                # 不加入 trainable_params（雖然有 hook 也可以加入，但沒必要）
            else:  # "train"
                B.requires_grad = True
                trainable_params.append(B)

    # 去重（避免重複加入）
    trainable_params = list(dict.fromkeys(trainable_params))
    print(f"DEBUG: Found {len(trainable_params)} trainable parameters.")
    print(f"DEBUG: Trainable parameter shapes: {[tuple(p.shape) for p in trainable_params]}")
    return trainable_params, hook_handles

# ================================================================
# 核心工具二：解凍所有參數
# ================================================================
def snapshot_params(policy: nn.Module):
    """抓取每個 Linear 的 weight/bias 張量拷貝，供前後比對。"""
    snap = {}
    for name, m in policy.named_modules():
        if isinstance(m, nn.Linear):
            snap[f"{name}.weight"] = m.weight.detach().clone()
            if m.bias is not None:
                snap[f"{name}.bias"] = m.bias.detach().clone()
    return snap

@torch.no_grad()
def diff_report_before_after(policy_before: nn.Module, policy_after: nn.Module, atol=1e-12):
    ok = True
    before_modules = dict(policy_before.named_modules())
    after_modules  = dict(policy_after.named_modules())

    print("\n=== Freeze 驗證報告（純淨塊應 ≈ 0）===")
    for name, m_after in after_modules.items():
        if not isinstance(m_after, nn.Linear):
            continue
        m_before = before_modules.get(name)
        if m_before is None:
            continue

        # 關鍵：統一到 CPU
        Wa = m_after.weight.detach().cpu()
        Wb = m_before.weight.detach().cpu()

        if Wa.shape != Wb.shape or Wa.ndim != 2:
            continue

        oh, ih = Wa.shape[0] // 2, Wa.shape[1] // 2
        if oh == 0 or ih == 0:
            continue

        d_pp = (Wa[:oh, :ih] - Wb[:oh, :ih]).abs().max().item()   # 父->父 左上（應≈0）
        d_pm = (Wa[:oh, ih:] - Wb[:oh, ih:]).abs().max().item()   # 父->母 右上（允許變）
        d_mp = (Wa[oh:, :ih] - Wb[oh:, :ih]).abs().max().item()   # 母->父 左下（允許變）
        d_mm = (Wa[oh:, ih:] - Wb[oh:, ih:]).abs().max().item()   # 母->母 右下（應≈0）

        if (m_after.bias is not None) and (m_before.bias is not None):
            ba = m_after.bias.detach().cpu()
            bb = m_before.bias.detach().cpu()
            if ba.numel() == 2 * oh:
                d_bt = (ba[:oh] - bb[:oh]).abs().max().item()
                d_bb = (ba[oh:] - bb[oh:]).abs().max().item()
            else:
                d_bt = d_bb = float('nan')
        else:
            d_bt = d_bb = float('nan')

        pure_ok = (d_pp <= atol) and (d_mm <= atol) and (np.isnan(d_bt) or d_bt <= atol) and (np.isnan(d_bb) or d_bb <= atol)
        ok &= pure_ok

        print(f"{'[OK]' if pure_ok else '[X ]'} {name:38s} "
              f"pure_pp:{d_pp:.2e}  pure_mm:{d_mm:.2e}  "
              f"cross_pm:{d_pm:.2e}  cross_mp:{d_mp:.2e}  "
              f"bias_top:{d_bt:.2e}  bias_bot:{d_bb:.2e}")

    print("✅ 純淨塊皆未更新（凍結生效）" if ok else "⚠️ 發現純淨塊有更新（凍結未生效）")
    return ok
def list_optimizer_params(optimizer: torch.optim.Optimizer):
    """列出目前 optimizer 管轄的參數形狀與名稱數量（粗略檢查）"""
    total = 0
    print("\n--- Optimizer 參數概覽 ---")
    for gi, g in enumerate(optimizer.param_groups):
        print(f"Group {gi} lr={g.get('lr')}")
        for p in g['params']:
            if p is None: 
                continue
            total += p.numel()
            print("   ", tuple(p.shape))
    print("總參數量(元素個數):", total)
def unfreeze_all(policy: nn.Module):
    """
    解凍模型中的所有參數，讓它們全部變為可訓練。

    注意：此函式無法移除已經註冊的梯度掛鉤。
    最穩健的做法是在呼叫此函式後，重新創建一個新的優化器。
    """
    print("\n--- 正在解凍所有模型參數 ---")
    for param in policy.parameters():
        param.requires_grad = True
    print("所有參數已解凍。")        
def run_episode(model, env: gym.Env, seed: int,save_video: bool = False,
    video_dir: str = "./videos",
    video_prefix: str = "episode",
    difficulty: float = None,) -> float:
    """
    在指定的環境和種子下，運行一個完整的 episode 並返回總獎勵。
    """
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        prefix = f"{video_prefix}_seed{seed}"
        if difficulty is not None:
            prefix += f"_diff{difficulty:.2f}"
        recorder = RecordVideo(
            env,
            video_folder=video_dir,
            name_prefix=prefix,
            episode_trigger=lambda ep: True,  # 每回合都錄
        )
        env = recorder
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
        obs = env.reset()
        
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    try:
        env.close()
        if recorder is not None:
            recorder.close()
    except Exception as e:
        print(f"⚠️ 關閉錄影環境時出錯: {e}")
    return total_reward

def compare_models():
    """
    主函式，用於載入並比較多個模型的性能，測試案例隨機抽樣自 hard_seeds.json。
    """
    # --- 1. 設定您要比較的模型 ---
    models_to_test = {
        "後代模型 (Evolved Child)": {
            "path": "./models/child_model5_x_model4_fused_full.pkl",
            "type": "pkl"
        },
        "父代模型 (Original Parent)": {
            "path": "./best_model/model4.zip",
            "type": "zip"
        
        },
        
        "母代模型 (Original Parent - Mom)": { # 👈 新增母代模型
        "path": "./best_model/model5.zip",
        "type": "zip"
        },

        

        
    }

    # --- 2. 設定測試環境與測試案例來源 ---
    env_id = "BipedalWalkerCustom-v0"
    hard_seeds_path = "./logs/hard_seeds.json"
    num_test_cases = 5000 # 👈 從 hard_seeds.json 中隨機抽樣的案例數量
    # 👈 新增難度過濾參數，與 sample_hardseeds 保持一致
    diff_min_filter = 0.0 
    diff_max_filter = 1.0

    print("="*50)
    print(f"正在開始模型性能比較...")
    print(f"將從 '{hard_seeds_path}' 隨機抽樣 {num_test_cases} 個測試案例。")
    print(f"難度範圍: [{diff_min_filter}, {diff_max_filter}]")
    print("="*50)

    # --- 3. 讀取並準備測試案例 (整合 sample_hardseeds 邏輯) ---
    if not os.path.exists(hard_seeds_path):
        print(f"❌ 錯誤: 找不到測試案例檔案 -> {hard_seeds_path}")
        return
        
    try:
        with open(hard_seeds_path, "r") as f:
            hardseed_dict = json.load(f)

        all_pairs = []
        for diff_str, seeds in hardseed_dict.items():
            d = float(diff_str)
            if diff_min_filter <= d <= diff_max_filter:
                for seed_str in seeds.keys():
                    all_pairs.append((d, int(seed_str)))

        # 確保抽樣數量不超過可用總數
        num_to_sample = min(num_test_cases, len(all_pairs))
        if num_to_sample == 0:
            print("❌ 錯誤: 在指定的難度範圍內找不到任何可用的測試案例。")
            return
            
        test_cases = random.sample(all_pairs, num_to_sample)
        print(f"成功抽樣 {len(test_cases)} 個測試案例。")

    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ 錯誤: 無法解析 '{hard_seeds_path}'。請確認檔案格式是否為巢狀字典。錯誤訊息: {e}")
        return

    # --- 4. 載入所有模型到記憶體 ---
    loaded_models = {}
    print("\n正在預先載入所有模型...")
    temp_env = gym.make(env_id)
    for model_name, model_info in models_to_test.items():
        model_path = model_info["path"]
        model_type = model_info["type"]
        
        if not os.path.exists(model_path):
            print(f"  - ❌ 警告: 找不到模型檔案 -> {model_path}。將跳過此模型。")
            continue

        model = None
        if model_type == "zip":
            model = PPO.load(model_path, env=temp_env)
        elif model_type == "pkl":
            with open(model_path, "rb") as f:
                model = cloudpickle.load(f)
        
        if model:
            loaded_models[model_name] = model
            print(f"  - ✅ 已載入: {model_name}")
    temp_env.close()

    if not loaded_models:
        print("\n❌ 錯誤: 沒有成功載入任何模型，測試中止。")
        return
    os.makedirs("./results", exist_ok=True)
    csv_path = "./results/model_comparison_results.csv"
    # --- 5. 逐一執行測試案例 ---
    all_results = {model_name: [] for model_name in loaded_models.keys()}
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["case_idx", "difficulty", "seed", "model_name", "reward"])
        for i, (difficulty, seed) in enumerate(test_cases):
            print(f"\n--- 測試案例 {i+1}/{len(test_cases)} (Difficulty: {difficulty}, Seed: {seed}) ---")
            
            # env = gym.make(env_id, difficulty=difficulty,render_mode="rgb_array")
            env = gym.make(env_id, difficulty=difficulty)
            for model_name, model in loaded_models.items():
                model.set_env(env)
                reward = run_episode(model, env, seed,save_video=False, video_dir = "./videos",video_prefix= model_name,difficulty=1.0)
                print(f"  - 模型 '{model_name}' 的獎勵: {reward:.2f}")
                all_results[model_name].append(reward)
                writer.writerow([i, difficulty, seed, model_name, reward])
                fcsv.flush()
            env.close()

    # --- 6. 顯示最終的比較報告 ---
    print("\n\n" + "="*50)
    print("📊 最終性能比較報告")
    print("="*50)
    for model_name, rewards in all_results.items():
        if not rewards: continue
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"模型: {model_name}")
        print(f"  - 平均獎勵: {avg_reward:.2f} ± {std_reward:.2f} (基於 {len(rewards)} 次測試)")
        print(f"  - 詳細分數: {[f'{r:.2f}' for r in rewards]}")
        print("-" * 20)
    stats = []
    for name, rewards in all_results.items():
        if rewards:
            stats.append((name, float(np.mean(rewards)), float(np.std(rewards)), len(rewards)))

    if not stats:
        print("⚠️ 無可視化資料（all_results 為空），跳過繪圖。")
        return

    # 依平均獎勵由高到低排序
    stats.sort(key=lambda x: x[1], reverse=True)
    labels = [s[0] for s in stats]
    means = [s[1] for s in stats]
    stds  = [s[2] for s in stats]
    ns    = [s[3] for s in stats]

    # 繪圖
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5)

    ax.set_title("Model Comparison — Average Reward (±1 SD)")
    ax.set_ylabel("Average Episode Reward")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 在每根柱上方標註 N（測試次數）
    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m, f"n={n}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_path = "./results/model_comparison_bar.png"
    plt.savefig(fig_path, dpi=150)
    print(f"📈 已輸出長條圖：{fig_path}")
def objective(trial: optuna.Trial) -> float:
    """
    Optuna 的目標函式。每一次呼叫，都代表一次完整的端到端實驗。
    """
    print(f"\n\n===== 開始 Optuna Trial #{trial.number} =====")
    
    # --- A. 讓 Optuna 為我們「建議」超參數 ---
    # 這裡只選擇了幾個最重要的作為範例，您可以增加更多
    
    # 離線蒸餾相關
    offline_lr = trial.suggest_float("offline_lr", 1e-6, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 1.0)
    temperature = trial.suggest_float("temperature", 1.0, 10.0)
    offline_epochs = trial.suggest_int("offline_epochs", 1, 20)

    # 線上微調相關
    online_lr = trial.suggest_float("online_lr", 1e-7, 1e-3, log=True)
    pure_lr_scale_factor = trial.suggest_float("pure_lr_scale_factor", 0.001, 0.5, log=True)
    finetune_steps = 100_000 # 為了快速迭代，可以先用較少的步數

    print("本次 Trial 的超參數:")
    print(trial.params)

    # --- B. 執行您完整的融合與訓練流程 ---
    try:
        model_folder = "./best_model"
        fitness_top  = "./logs/ga_eval1/fitness_top.json"
        distill_pt   = "./logs/ga_eval/mtkd_continuous.pt(1)"
        env_id       = "BipedalWalkerCustom-v0"

        with open(fitness_top, "r", encoding="utf-8") as f:
            table = json.load(f)
        names = list(table.keys())
        if len(names) < 2:
            raise ValueError("fitness_top.json 至少需要兩個模型")
        
        # mom_name, dad_name = random.sample(names, 2)
        mom_name, dad_name = "model4", "model5"
        
        # resolve_path 函式保持不變
        def resolve_path(name: str, root: str) -> str:
            # ... (您提供的程式碼，無需修改) ...
            if os.path.isabs(name) and os.path.exists(name): return name
            candidates = []
            if name.endswith(".zip"): candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))
            for p in candidates:
                if os.path.exists(p): return p
             # 應該加上找不到的處理
            raise FileNotFoundError(f"在 '{root}' 資料夾中找不到模型檔案 '{name}' (或 .zip)")

        mom_path = resolve_path(mom_name, model_folder)
        dad_path = resolve_path(dad_name, model_folder)

        print(f"Parents -> mom: {mom_name} ({mom_path}), dad: {dad_name} ({dad_path})")

        env = gym.make("BipedalWalkerCustom-v0", difficulty=0.5)

        mom = PPO.load(mom_path, env=env)
        dad = PPO.load(dad_path, env=env)
        # 2.1 建構全新的「雙子網路」策略
        child_policy = create_dual_channel_policy(dad.policy, mom.policy)
        # 2.2 對新網路執行交叉通道歸零
        zero_initialize_crosstalk(child_policy)
        # 2.3 創建一個新的 PPO Agent 來承載我們的子代策略
        # 注意：這裡的超參數應該與您的 dad/mom agent 保持一致
    
        child_agent = PPO(
            policy=dad.policy.__class__,
            env=env,
            learning_rate=online_lr, # 線上微調時的學習率
            n_steps=dad.n_steps,
            batch_size=dad.batch_size,
            n_epochs=dad.n_epochs,
            gamma=dad.gamma,
            gae_lambda=dad.gae_lambda,
            clip_range=dad.clip_range,
            ent_coef=dad.ent_coef,
            verbose=0
        )
        child_agent.policy = child_policy.to(child_agent.device)
        
        # 2.4 【關鍵】為子代策略設定我們特製的「精細版差異化優化器」
        # 注意：線上微調時，我們可能希望交叉通道的學習率高一些，所以這裡用了 lr_pure/10
        child_agent.policy.optimizer = create_refined_differential_optimizer(
            policy=child_agent.policy,
            lr_pure=child_agent.learning_rate, # 使用 PPO agent 的主學習率
            cross_lr_scale_factor=pure_lr_scale_factor # 線上微調時，給予交叉通道 10% 的學習率
        )
        
        # 2.5 執行「全局離線蒸餾」作為熱身 (使用軟硬結合的損失函數)
        print("\n--- 開始執行離線蒸餾預訓練 ---")
        train_network_offline(
            ppo_model=child_agent,
            pt_path=distill_pt,
            epochs=offline_epochs,             # 離線訓練10輪
            lr=offline_lr,               # 離線訓練使用稍大的學習率
            vf_coef=0.5,
            temperature=temperature,       # 軟目標的溫度
            alpha=alpha,             # 軟目標的權重
            device=child_agent.device
        )
        
        # 2.6 執行最終的「線上微調」
        print("\n--- 開始執行線上強化學習微調 ---")
        finetune_steps = 100000
        auto_difficulty_callback = AutoDifficultyCallback(
        env,None, eval_freq=10000, reward_threshold=250, increase=0.05, verbose=1,shared_flags=False,cooldown_steps=0,hardseed_save_path="./logs/hard_seeds.json"
    )
        child_agent.learn(total_timesteps=finetune_steps, callback=[ auto_difficulty_callback,],progress_bar=True)
        # --- C. 評估最終性能並返回分數 ---
        # 為了節省時間，可以用較少的 episode 進行評估
        mean_reward, std_reward = evaluate_policy(child_agent, env, n_eval_episodes=20)
        final_difficulty = env.unwrapped.difficulty
        print(f"Trial #{trial.number} 完成，平均獎勵: {mean_reward}")
        difficulty_bonus = 150.0
        composite_score = mean_reward + 250*((final_difficulty-0.5)/0.05)
        env.close()
        return composite_score

    except Exception as e:
        print(f"Trial #{trial.number} 因錯誤而失敗: {e}")
        # 告知 Optuna 這次嘗試失敗了
        print("--- DETAILED TRACEBACK ---")
        traceback.print_exc()
        print("---------")
        return -1000.0 # 返回一個極差的分數
def analyze_heatmaps(policies: dict, save_dir: str):
    """
    為多個模型的每一層權重繪製熱力圖，並並排儲存以便比較。
    """
    print("\n--- 正在生成權重熱力圖 ---")
    # 獲取所有模型共享的層名稱
    sample_policy = next(iter(policies.values()))
    layer_names = [name for name, module in sample_policy.named_modules() if isinstance(module, nn.Linear)]

    with torch.no_grad():
        for name in layer_names:
            num_models = len(policies)
            fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
            if num_models == 1: axes = [axes] # 處理只有一個模型的情況

            for i, (model_name, policy) in enumerate(policies.items()):
                module = dict(policy.named_modules())[name]
                W = module.weight.data.cpu().numpy()
                ax = axes[i]
                
                sns.heatmap(W, cmap='coolwarm', center=0.0, ax=ax, cbar=i==num_models-1)
                ax.set_title(f"{model_name}\nShape: {W.shape}", fontsize=14)
                
                # 如果是子代模型，畫出輔助線
                if "child" in model_name.lower():
                    out_half, in_half = W.shape[0] // 2, W.shape[1] // 2
                    if out_half > 0: ax.axhline(out_half, color='black', linewidth=2.5)
                    if in_half > 0: ax.axvline(in_half, color='black', linewidth=2.5)

            fig.suptitle(f"Weighted Heatmap Comparison: Layer '{name}'", fontsize=20, fontweight='bold')
            plt.rcParams['axes.unicode_minus'] = False
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            save_path = os.path.join(save_dir, f"heatmap_{name.replace('.', '_')}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"  - 已儲存 '{name}' 層的熱力圖比較至: {save_path}")
def analyze_weight_difference(child_policy: nn.Module, dad_policy: nn.Module, mom_policy: nn.Module, save_dir: str):
    """
    計算並視覺化子代模型權重與其親代對應部分之間的差異。
    上半部分與父親比較，下半部分與母親比較。
    """
    print("\n--- 正在生成權重【差異】熱力圖 ---")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for name, child_module in child_policy.named_modules():
            if isinstance(child_module, nn.Linear):
                dad_module = dict(dad_policy.named_modules()).get(name)
                mom_module = dict(mom_policy.named_modules()).get(name)
                
                # 確保父母雙方都有對應的層
                if not (dad_module and mom_module): 
                    print(f"  - 跳過層 '{name}': 在父代或母代中找不到對應層。")
                    continue
                
                Wc = child_module.weight.data
                Wd = dad_module.weight.data
                Wm = mom_module.weight.data

                # 檢查維度是否匹配，以及是否可分割
                if Wc.shape != Wd.shape or Wc.shape != Wm.shape:
                    print(f"  - 跳過層 '{name}': 維度不匹配。")
                    continue
                
                out_half = Wc.shape[0] // 2
                if out_half == 0 or Wc.shape[0] % 2 != 0:
                    print(f"  - 跳過層 '{name}': 輸出維度無法被對半分。")
                    continue

                # --- 核心計算邏輯 ---
                # 計算上半部分的差異 (Child - Dad)
                diff_top = Wc[:out_half, :] - Wd[:out_half, :]
                
                # 計算下半部分的差異 (Child - Mom)
                diff_bottom = Wc[out_half:, :] - Wm[out_half:, :]
                
                # 將兩個差異區塊拼接回來，形成一個完整的差異矩陣
                diff_matrix = torch.cat([diff_top, diff_bottom], dim=0).cpu().numpy()
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                # --- 繪圖 ---
                plt.figure(figsize=(10, 8))
                
                # 找到差異的最大絕對值，用於設定對稱的顏色標尺
                vmax = np.abs(diff_matrix).max()
                
                # 使用熱力圖視覺化差異矩陣
                sns.heatmap(diff_matrix, cmap='coolwarm', center=0.0, vmin=-vmax, vmax=vmax)
                
                # 繪製輔助線，標示出不同的比較區域
                ax = plt.gca()
                ax.axhline(out_half, color='black', linewidth=2.5)

                ax.set_title(f"權重差異圖 (訓練後變化): Layer '{name}'\n上半部: Child - Dad | 下半部: Child - Mom", fontsize=16)
                plt.xlabel("輸入特徵 (Input Features)")
                plt.ylabel("輸出神經元 (Output Neurons)")
                plt.tight_layout()
                
                # --- 儲存圖表 ---
                save_path = os.path.join(save_dir, f"difference_heatmap_{name.replace('.', '_')}.png")
                plt.savefig(save_path, dpi=150)
                plt.close() # 關閉圖表，釋放記憶體
                print(f"  - 已儲存 '{name}' 層的差異圖至: {save_path}")
def analyze_histograms(policy: nn.Module, save_dir: str, model_name: str):
    """
    為指定模型的每一層權重，繪製四個通道的權重分佈直方圖。
    """
    print(f"\n--- 正在為 '{model_name}' 生成權重分佈直方圖 ---")
    with torch.no_grad():
        for name, module in policy.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                out_half, in_half = W.shape[0] // 2, W.shape[1] // 2
                
                if in_half == 0 or out_half == 0: continue # 跳過無法分割的層

                blocks = {
                    "父->父 (左上)": W[:out_half, :in_half].flatten().cpu().numpy(),
                    "母->父 (右上)": W[:out_half, in_half:].flatten().cpu().numpy(),
                    "父->母 (左下)": W[out_half:, :in_half].flatten().cpu().numpy(),
                    "母->母 (右下)": W[out_half:, in_half:].flatten().cpu().numpy(),
                }
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f"'{model_name}' - Layer '{name}' 權重分佈", fontsize=20, fontweight='bold')
                
                for ax, (title, data) in zip(axes.flatten(), blocks.items()):
                    sns.histplot(data, bins=50, kde=True, ax=ax)
                    ax.set_title(title)
                    ax.set_xlabel("權重值")
                    ax.set_ylabel("數量")
                plt.rcParams['axes.unicode_minus'] = False
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                save_path = os.path.join(save_dir, f"histogram_{name.replace('.', '_')}.png")
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"  - 已儲存 '{name}' 層的直方圖至: {save_path}")
def collect_actions(model: PPO, env: gym.Env, seed: int) -> np.ndarray:
    """在指定的環境和種子碼下，收集模型完整的動作序列。"""
    actions = []
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        # 使用確定性動作，以確保策略是固定的
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.array(actions)
def collect_trajectories(model: PPO, env: gym.Env, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """在指定的環境和種子碼下，收集模型完整的【狀態】和【動作】序列。"""
    observations = []
    actions = []
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        observations.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.array(observations), np.array(actions)
def plot_action_distributions(actions_dict: dict, save_dir: str, difficulty: float, seed: int):
    """
    為多個模型的動作序列，繪製每個動作維度的分佈直方圖/密度圖。
    """
    print(f"\n✅ 正在繪製動作分佈直方圖...")
    
    # 獲取動作維度
    sample_actions = next(iter(actions_dict.values()))
    action_dim = sample_actions.shape[1]

    for i in range(action_dim):
        plt.figure(figsize=(12, 8))
        
        # 為每個模型繪製核密度估計圖 (KDE Plot)，更平滑
        for model_name, actions in actions_dict.items():
            sns.kdeplot(actions[:, i], label=model_name, fill=True, alpha=0.5)

        plt.title(f'動作維度 {i+1} 的分佈比較\n(Difficulty: {difficulty}, Seed: {seed})', fontsize=16)
        plt.xlabel(f'動作 {i+1} 的數值', fontsize=12)
        plt.ylabel('密度 (Density)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"action_distribution_dim_{i+1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   - 已儲存動作維度 {i+1} 的分佈圖至: {save_path}")
def run_state_conditioned_analysis():
    """
    執行完整的、基於物理特徵的狀態條件下的動作分析流程。
    """
    print("="*60)
    print("      模型在特定物理狀態下的動作對比分析")
    print("="*60)
    env_id = "BipedalWalkerCustom-v0"
    best_difficulty=0.8
    best_seed=3729259198
    conditions = {
        "高速前進時 (vel_x > 1.0)": 
            lambda obs: obs[2] > 1.0,
        "身體失衡後仰時 (hull_ang_vel < -0.3)": 
            lambda obs: obs[1] < -0.3,
        "處於滯空/下墜狀態時 (vel_y < -0.5)":
            lambda obs: obs[3] < -0.5,
        "左腿大步邁出時 (hip_joint_1_angle < -0.3)":
            lambda obs: obs[4] < -0.3,
        "即將撞上前方障礙物時 (lidar[0] < 0.5)":
            lambda obs: obs[14] < 0.5, # lidar[0] 是正前方的雷達
    }
    print("已定義的關鍵物理情境:")
    for name in conditions.keys():
        print(f"  - {name}")

    # --- 2. 載入模型並收集狀態-動作軌跡 ---
    env = gym.make(env_id, difficulty=best_difficulty)
    
    

    print(f"\n--- 正在收集軌跡數據 (Difficulty: {best_difficulty}, Seed: {best_seed}) ---")
    models_to_test = {
        "Child_model (Evolved Child)": {
            "path": "./models/child_model2_x_model3_fused_full(5).pkl",
            "type": "pkl"
        },
        "Dad_model (Original Parent-Dad)": {
            "path": "./best_model/model2.zip",
            "type": "zip"
        
        },
        
        "Mom_model (Original Parent - Mom)": { # 👈 新增母代模型
        "path": "./best_model/model3.zip",
        "type": "zip"
        },

        

        
    }
    
    
    # --- 2. 設定測試環境與測試案例來源 ---
    save_dir = "./actions_stept"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    loaded_models = {}
    print("\n正在預先載入所有模型...")
    temp_env = gym.make(env_id)
    for model_name, model_info in models_to_test.items():
            model_path = model_info["path"]
            model_type = model_info["type"]
            
            if not os.path.exists(model_path):
                print(f"  - ❌ 警告: 找不到模型檔案 -> {model_path}。將跳過此模型。")
                continue

            model = None
            if model_type == "zip":
                model = PPO.load(model_path, env=temp_env)
            elif model_type == "pkl":
                with open(model_path, "rb") as f:
                    model = cloudpickle.load(f)
            
            if model:
                loaded_models[model_name] = model
                print(f"  - ✅ 已載入: {model_name}")
    temp_env.close()
    trajectories = {}
    for name, model in loaded_models.items():
        trajectories[name] = collect_trajectories(model, env, best_seed)
    
    min_len = min(len(obs) for obs, act in trajectories.values())
    
    # --- 3. 逐一分析每個情境 ---
    os.makedirs(save_dir, exist_ok=True)
    
    for cond_name, cond_func in conditions.items():
        print(f"\n--- 正在分析情境: {cond_name} ---")
        
        child_obs, _ = trajectories["Child_model (Evolved Child)"]
        condition_indices = [i for i, obs in enumerate(child_obs[:min_len]) if cond_func(obs)]

        if not condition_indices:
            print("  -> 在此回合中，子代模型未觸發此情境。")
            continue
            
        print(f"  -> 在 {min_len} 幀中，共找到 {len(condition_indices)} 個符合條件的時間點。")

        action_data = []
        for model_name, (obs_seq, act_seq) in trajectories.items():
            for idx in condition_indices:
                action = act_seq[idx]
                for dim in range(action.shape[0]):
                    action_data.append({
                        "模型": model_name,
                        "動作維度": f"動作 {dim+1}",
                        "動作值": action[dim]
                    })
        
        df_actions = pd.DataFrame(action_data)

        # --- 4. 繪製並儲存該情境下的動作比較圖 ---
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure(figsize=(15, 8))
        
        sns.boxplot(data=df_actions, x="動作維度", y="動作值", hue="模型", palette="pastel")
        sns.stripplot(data=df_actions, x="動作維度", y="動作值", hue="模型", dodge=True, jitter=0.2, palette="deep", size=4)

        plt.title(f'關鍵情境下的動作比較: {cond_name}', fontsize=18)
        plt.ylabel("動作值 (Action Value)", fontsize=12)
        plt.xlabel("動作維度", fontsize=12)
        plt.legend(title='模型')
        plt.grid(True, linestyle='--')
        plt.tight_layout()

        safe_cond_name = "".join(c for c in cond_name if c.isalnum() or c in (' ', '_')).rstrip()
        save_path = os.path.join(save_dir, f"state_conditioned_{safe_cond_name.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> ✅ 已儲存分析圖表至: {save_path}")

    env.close()
    print("\n\n分析完成！")
def run_action_analysis():
    """
    執行完整的動作相似度分析流程。
    """
    print("="*60)
    print("                模型動作相似度分析")
    print("="*60)

    env_id = "BipedalWalkerCustom-v0"
    # --- 2. 載入模型並收集動作序列 ---
    env = gym.make(env_id, difficulty=0.70)
    best_seed=587663355
    models_to_test = {
        "Child_model (Evolved Child)": {
            "path": "./models/child_model5_x_model4_fused_full.pkl",
            "type": "pkl"
        },
        "Dad_model (Original Parent-Dad)": {
            "path": "./best_model/model4.zip",
            "type": "zip"
        
        },
        
        "Mom_model (Original Parent - Mom)": { # 👈 新增母代模型
        "path": "./best_model/model5.zip",
        "type": "zip"
        },

        

        
    }
    best_difficulty=0.85
    # --- 2. 設定測試環境與測試案例來源 ---
    save_dir = "./actions_analysis1"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    loaded_models = {}
    print("\n正在預先載入所有模型...")
    temp_env = gym.make(env_id)
    for model_name, model_info in models_to_test.items():
            model_path = model_info["path"]
            model_type = model_info["type"]
            
            if not os.path.exists(model_path):
                print(f"  - ❌ 警告: 找不到模型檔案 -> {model_path}。將跳過此模型。")
                continue

            model = None
            if model_type == "zip":
                model = PPO.load(model_path, env=temp_env)
            elif model_type == "pkl":
                with open(model_path, "rb") as f:
                    model = cloudpickle.load(f)
            
            if model:
                loaded_models[model_name] = model
                print(f"  - ✅ 已載入: {model_name}")
    temp_env.close()
    loaded_policies = {name: agent.policy for name, agent in loaded_models.items()}
    child_policy = loaded_policies.get("Child_model (Evolved Child)")
    mom_policy = loaded_policies.get( "Mom_model (Original Parent - Mom)")
    dad_policy = loaded_policies.get( "Dad_model (Original Parent-Dad)")
    print("\n--- 正在收集動作序列... ---")
    actions_dad = collect_actions(dad_policy, env, best_seed)
    actions_mom = collect_actions(mom_policy, env, best_seed)
    actions_child = collect_actions(child_policy, env, best_seed)
    
    # 確保長度一致以便比較
    min_len = min(len(actions_dad), len(actions_mom), len(actions_child))
    actions_dad = actions_dad[:min_len]
    actions_mom = actions_mom[:min_len]
    actions_child = actions_child[:min_len]
    print(f"動作序列收集完成，已對齊長度至 {min_len} 幀。")

    # --- 3. 計算並報告動作相似度 ---
    # 使用 L2 距離 (歐氏距離的平方) 的平均值，即 MSE
    mse_child_dad = np.mean((actions_child - actions_dad)**2)
    mse_child_mom = np.mean((actions_child - actions_mom)**2)

    print("\n\n" + "="*60)
    print("📊 動作相似度分析報告 (均方誤差 MSE，越低越相似)")
    print("="*60)
    print(f"  - 子代 vs. 父代 (Child vs. Dad): {mse_child_dad:.6f}")
    print(f"  - 子代 vs. 母代 (Child vs. Mom): {mse_child_mom:.6f}")
    
    if mse_child_dad < mse_child_mom:
        print("\n【結論】: 在此案例中，子代的行為模式更接近【父代】。")
    elif mse_child_mom < mse_child_dad:
        print("\n【結論】: 在此案例中，子代的行為模式更接近【母代】。")
    else:
        print("\n【結論】: 在此案例中，子代與父母的行為相似度幾乎相同。")


    # --- 4. 視覺化分析 ---
    os.makedirs(save_dir, exist_ok=True)
    
    # 4a. 繪製總體相似度長條圖
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['vs. 父代 (Dad)', 'vs. 母代 (Mom)'], y=[mse_child_dad, mse_child_mom], palette='pastel')
    plt.title('子代與父母的動作策略相似度', fontsize=16)
    plt.ylabel('平均均方誤差 (MSE)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "action_similarity_mse.png"), dpi=300)
    print(f"\n✅ 已儲存總體相似度長條圖至 '{save_dir}/action_similarity_mse.png'")

    # 4b. 繪製逐幀動作對比曲線圖
    action_dim = actions_child.shape[1]
    timesteps = np.arange(min_len)
    
    fig, axes = plt.subplots(action_dim, 1, figsize=(18, 5 * action_dim), sharex=True)
    if action_dim == 1: axes = [axes]
        
    fig.suptitle(f'逐幀動作對比 (Difficulty: {best_difficulty}, Seed: {best_seed})', fontsize=20, y=0.97)

    for i in range(action_dim):
        axes[i].plot(timesteps, actions_dad[:, i], label='父代 (Dad)', color='royalblue', alpha=0.7, linewidth=1.5)
        axes[i].plot(timesteps, actions_mom[:, i], label='母代 (Mom)', color='crimson', alpha=0.7, linewidth=1.5)
        axes[i].plot(timesteps, actions_child[:, i], label='子代 (Child)', color='forestgreen', linewidth=2.5)
        axes[i].set_ylabel(f'動作維度 {i+1}')
        axes[i].legend()
        axes[i].grid(True, linestyle='--')

    axes[-1].set_xlabel('時間步 (Timestep)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "action_trajectory_comparison.png"), dpi=300)
    print(f"✅ 已儲存逐幀動作對比曲線圖至 '{save_dir}/action_trajectory_comparison.png'")
    actions_dict = {
        "父代 (Dad)": actions_dad,
        "母代 (Mom)": actions_mom,
        "子代 (Child)": actions_child
    }
    plot_action_distributions(actions_dict, save_dir, best_difficulty, best_seed)
    
    env.close()
    print("\n分析完成！所有圖表已儲存至:", save_dir)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test",  action="store_true")
    parser.add_argument("--prune-hardseeds", action="store_true")
    parser.add_argument("--rename_modelname", action="store_true")
    parser.add_argument("--test_population", action="store_true")
    parser.add_argument("--evolved", action="store_true")
    parser.add_argument("--evolved_all", action="store_true")
    parser.add_argument("--compare_test", action="store_true")
    parser.add_argument("--find_value", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--weight", action="store_true")
    parser.add_argument("--action", action="store_true")
    parser.add_argument("--action_state", action="store_true")
    args = parser.parse_args()

    if args.train:
        more_train()
    elif args.test:
        test()
    elif args.compare_test:
        compare_models()
    elif args.prune_hardseeds:
        prune_hardseeds()
    elif args.rename_modelname:
        rename_modelname()
    elif args.test_population:
        model_folder = "./best_model/run_4"
        model_files = sorted([os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith(".zip")])
        # 評估並儲存資料
        fitness_out, top_names = test_population(
        model_paths=model_files,
        hardseed_path="./logs/hard_seeds.json",
        env_id="BipedalWalkerCustom-v0",
        num_samples=300,
        eval_n_stage1=1,
        eval_n_stage2=10,
        reward_threshold=250,
        top_k=30,                      # 你要的 Top-30
        save_dir="./logs/ga_eval1",
        save_name="mtkd_continuous.pt(1)",
        num_workers=10,
        device_str="cpu",              # 想用 GPU 改成 "cuda"
        diff_min=0.0,
        diff_max=1.0,
    )
    elif args.action:
        run_action_analysis()
    elif args.action_state:
        run_state_conditioned_analysis()
    elif args.weight:
        models_to_test = {
        "Child_model (Evolved Child)": {
            "path": "./models/child_model2_x_model3_fused_full(5).pkl",
            "type": "pkl"
        },
        "Dad_model (Original Parent-Dad)": {
            "path": "./best_model/model2.zip",
            "type": "zip"
        
        },
        
        "Mom_model (Original Parent - Mom)": { # 👈 新增母代模型
        "path": "./best_model/model3.zip",
        "type": "zip"
        },

        

        
    }

    # --- 2. 設定測試環境與測試案例來源 ---
        env_id = "BipedalWalkerCustom-v0"
        save_dir = "./weight_analysis"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        loaded_models = {}
        print("\n正在預先載入所有模型...")
        temp_env = gym.make(env_id)
        for model_name, model_info in models_to_test.items():
            model_path = model_info["path"]
            model_type = model_info["type"]
            
            if not os.path.exists(model_path):
                print(f"  - ❌ 警告: 找不到模型檔案 -> {model_path}。將跳過此模型。")
                continue

            model = None
            if model_type == "zip":
                model = PPO.load(model_path, env=temp_env)
            elif model_type == "pkl":
                with open(model_path, "rb") as f:
                    model = cloudpickle.load(f)
            
            if model:
                loaded_models[model_name] = model
                print(f"  - ✅ 已載入: {model_name}")
        temp_env.close()
        loaded_policies = {name: agent.policy for name, agent in loaded_models.items()}
        child_policy = loaded_policies.get("Child_model (Evolved Child)")
        mom_policy = loaded_policies.get( "Mom_model (Original Parent - Mom)")
        dad_policy = loaded_policies.get("Dad_model (Original Parent-Dad)")
        analyze_heatmaps(loaded_policies, save_dir=os.path.join(save_dir, "heatmaps"))
        analyze_weight_difference(child_policy=child_policy,mom_policy=mom_policy,dad_policy=dad_policy,save_dir=save_dir)
    # 方法二：直方圖 (只對冠軍子代做)
        if child_policy:
            analyze_histograms(child_policy, save_dir=os.path.join(save_dir, "histograms"), model_name="Champion_Child")
        if mom_policy:
            analyze_histograms(mom_policy, save_dir=os.path.join(save_dir, "histograms_mom"), model_name="Mom")
        if dad_policy:
            analyze_histograms(dad_policy, save_dir=os.path.join(save_dir, "histograms_dad"), model_name="Dad")
        print("\n\n分析完成！所有圖表已儲存至:", save_dir)
    elif args.offline: 
        model_folder = "./best_model"
        fitness_top  = "./logs/ga_eval/fitness_top.json"
        distill_pt   = "./logs/ga_eval/mtkd_continuous.pt(2)"
        env_id       = "BipedalWalkerCustom-v0"
        with open(fitness_top, "r", encoding="utf-8") as f:
            table = json.load(f)
        names = list(table.keys())
        if len(names) < 2:
            raise ValueError("fitness_top.json 至少需要兩個模型")
        
        # mom_name, dad_name = random.sample(names, 2)
        mom_name, dad_name = "model3", "model2"
        
        # resolve_path 函式保持不變
        def resolve_path(name: str, root: str) -> str:
            # ... (您提供的程式碼，無需修改) ...
            if os.path.isabs(name) and os.path.exists(name): return name
            candidates = []
            if name.endswith(".zip"): candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))
            for p in candidates:
                if os.path.exists(p): return p
            return None # 應該加上找不到的處理

        mom_path = resolve_path(mom_name, model_folder)
        dad_path = resolve_path(dad_name, model_folder)

        print(f"Parents -> mom: {mom_name} ({mom_path}), dad: {dad_name} ({dad_path})")

        env = gym.make("BipedalWalkerCustom-v0", difficulty=0.5)

        mom = PPO.load(mom_path, env=env)
        dad = PPO.load(dad_path, env=env)
        child_agent = PPO(
            policy=dad.policy.__class__,
            env=env,
            learning_rate=4.567629084674937e-06, # 線上微調時的學習率
            n_steps=dad.n_steps,
            batch_size=dad.batch_size,
            n_epochs=dad.n_epochs,
            gamma=dad.gamma,
            gae_lambda=dad.gae_lambda,
            clip_range=dad.clip_range,
            ent_coef=dad.ent_coef,
            verbose=0
        )
        distill_lr = 1e-4 # 為蒸餾設定一個合理的學習率
        optimizer = torch.optim.Adam(child_agent.policy.parameters(), lr=distill_lr)
        train_network_offline(
            ppo_model=child_agent,
            pt_path=distill_pt,
            optimizer=optimizer,
            epochs=1,             # 離線訓練10輪
            lr= 1e-4,               # 離線訓練使用稍大的學習率
            vf_coef=0.5,
            temperature=4.414784424905544,       # 軟目標的溫度
            alpha=0.5962410908066522,             # 軟目標的權重
            device=child_agent.device
        )
        print("\n--- 開始執行線上強化學習微調 ---")
        finetune_steps = 100000
        auto_difficulty_callback = AutoDifficultyCallback(
        env,None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,shared_flags=False,cooldown_steps=0,hardseed_save_path="./logs/hard_seeds.json"
    )
        child_agent.learn(total_timesteps=finetune_steps, callback=[ auto_difficulty_callback,],progress_bar=True)

        evolved = child_agent # 將訓練完成的 agent 賦值給 evolved 變數
        
        # --- 步驟 3: 儲存訓練好的模型 (這部分保持不變) ---
        print("\n--- 正在儲存融合後的模型 ---")
        out_dir = Path("./models"); out_dir.mkdir(parents=True, exist_ok=True)
        dad_base = os.path.splitext(os.path.basename(dad_path))[0]
        mom_base = os.path.splitext(os.path.basename(mom_path))[0]
        tag = f"child_{dad_base}_x_{mom_base}_fused" # 加上 fused 標籤以區分
        zip_path = out_dir / f"{tag}.zip"
        evolved.save(str(zip_path))
        pkl_path = out_dir / f"{tag}_full.pkl"
        evolved.env = None # 移除 env 以便打包
        with open(pkl_path, "wb") as f:
            # 需要 import cloudpickle
            import cloudpickle
            cloudpickle.dump(evolved, f)

        print("✅ Saved:")
        print(f"   - SB3 zip:   {zip_path}")
        print(f"   - Full .pkl: {pkl_path}")    
    elif args.evolved: 
        model_folder = "./best_model"
        fitness_top  = "./logs/ga_eval/fitness_top.json"
        distill_pt   = "./logs/ga_eval1/mtkd_continuous.pt(1)"
        env_id       = "BipedalWalkerCustom-v0"

        # 讀榜單並抽兩個不同模型名
        with open(fitness_top, "r", encoding="utf-8") as f:
            table = json.load(f)
        names = list(table.keys())
        if len(names) < 2:
            raise ValueError("fitness_top.json 至少需要兩個模型")
        #mom_name, dad_name = random.sample(names, 2)
        mom_name, dad_name = "model4","model5"
        # 把名稱解析成實際檔案路徑
        def resolve_path(name: str, root: str) -> str:
            # 絕對路徑直接用
            if os.path.isabs(name) and os.path.exists(name):
                return name

            candidates = []
            if name.endswith(".zip"):
                candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))

            for p in candidates:
                if os.path.exists(p):
                    return p

            # 最後保險：掃描資料夾，把去副檔名後的名字比對

        mom_path = resolve_path(mom_name, model_folder)
        dad_path = resolve_path(dad_name, model_folder)

        print(f" Parents -> mom: {mom_name} ({mom_path}), dad: {dad_name} ({dad_path})")

        # 建環境（要傳環境物件給 PPO，不是字串）
        env = gym.make("BipedalWalkerCustom-v0", difficulty=0.0)

        # 載入父母
        mom = PPO.load(mom_path, env=env)  # 可加 device="cpu"/"cuda"
        dad = PPO.load(dad_path, env=env)
        debug_plan = {
        1: ("mlp_extractor.policy_net", 0),                # 第 1 輪：強制交換整個 action_net
      # 第 2 輪：強制交換 value 主幹的第 2 層
        2: ("action_net", -1), 
        }
        print("  DEBUG MODE: Running with a predefined evolution plan:")
        for round_num, action in debug_plan.items():
            print(f"  - Round {round_num}: Crossover unit -> {action[0]} (idx={action[1]})")
        # 交配 + progressive
        evolved = progressive_evolve(
            dad, mom, env, distill_pt=distill_pt,debug_plan=debug_plan
        )

        # 輸出檔名：child_父_x_母.zip
        out_dir = Path("./models"); out_dir.mkdir(parents=True, exist_ok=True)
        dad_base = os.path.splitext(os.path.basename(dad_path))[0]
        mom_base = os.path.splitext(os.path.basename(mom_path))[0]
        tag = f"child_{dad_base}_x_{mom_base}"

        # A) SB3 官方格式（最穩，推薦）
        zip_path = out_dir / f"{tag}.zip"
        evolved.save(str(zip_path))

        # B) 輕量化：只存 policy 權重（GA/蒸餾方便）

        # C) 最後手段：完整 pickle（移除 env，避免把環境也打包）
        pkl_path = out_dir / f"{tag}_full.pkl"
        evolved.env = None
        with open(pkl_path, "wb") as f:
            cloudpickle.dump(evolved, f)

        print("✅ Saved:")
        print(f"   - SB3 zip:      {zip_path}")
        print(f"   - Full .pkl:    {pkl_path}")
    elif args.evolved_all:
    # --- 步驟 1: 參數設定與父母模型載入 (這部分保持不變) ---
        model_folder = "./best_model"
        fitness_top  = "./logs/ga_eval/fitness_top.json"
        distill_pt   = "./logs/ga_eval1/mtkd_continuous.pt(1)"
        env_id       = "BipedalWalkerCustom-v0"

        with open(fitness_top, "r", encoding="utf-8") as f:
            table = json.load(f)
        names = list(table.keys())
        if len(names) < 2:
            raise ValueError("fitness_top.json 至少需要兩個模型")
        
        # mom_name, dad_name = random.sample(names, 2)
        mom_name, dad_name = "model4", "model5"
        
        # resolve_path 函式保持不變
        def resolve_path(name: str, root: str) -> str:
            # ... (您提供的程式碼，無需修改) ...
            if os.path.isabs(name) and os.path.exists(name): return name
            candidates = []
            if name.endswith(".zip"): candidates.append(os.path.join(root, name))
            else:
                candidates.append(os.path.join(root, name + ".zip"))
                candidates.append(os.path.join(root, name))
            for p in candidates:
                if os.path.exists(p): return p
            return None # 應該加上找不到的處理

        mom_path = resolve_path(mom_name, model_folder)
        dad_path = resolve_path(dad_name, model_folder)

        print(f"Parents -> mom: {mom_name} ({mom_path}), dad: {dad_name} ({dad_path})")

        env = gym.make("BipedalWalkerCustom-v0", difficulty=0.5)

        mom = PPO.load(mom_path, env=env)
        dad = PPO.load(dad_path, env=env)
        # 2.1 建構全新的「雙子網路」策略
        child_policy = create_dual_channel_policy(dad.policy, mom.policy)
       
        # 2.2 對新網路執行交叉通道歸零
        zero_initialize_crosstalk(child_policy)
        # 2.3 創建一個新的 PPO Agent 來承載我們的子代策略
        # 注意：這裡的超參數應該與您的 dad/mom agent 保持一致
        child_agent = PPO(
            policy=dad.policy.__class__,
            env=env,
            learning_rate=4.567629084674937e-06, # 線上微調時的學習率
            n_steps=dad.n_steps,
            batch_size=dad.batch_size,
            n_epochs=dad.n_epochs,
            gamma=dad.gamma,
            gae_lambda=dad.gae_lambda,
            clip_range=dad.clip_range,
            ent_coef=dad.ent_coef,
            verbose=0
        )
        child_agent.policy = child_policy.to(child_agent.device)
        trainable_params, hook_handles = freeze_pure_channels(child_agent.policy,bias_mode="freeze",allow_train_if_unsplit=False)
        child_agent.policy.optimizer = torch.optim.Adam(trainable_params, lr=0.008780571292219902,weight_decay=0.0)
        list_optimizer_params(child_agent.policy.optimizer)
        # distill_lr = 1e-4 # 為蒸餾設定一個合理的學習率
        # child_agent.policy.optimizer = torch.optim.Adam(child_agent.policy.parameters(), lr=distill_lr)

# 2) 先拍快照
        policy_before = copy.deepcopy(child_agent.policy).cpu()
        print("\n--- 開始執行離線蒸餾預訓練 ---")
        train_network_offline(
            ppo_model=child_agent,
            pt_path=distill_pt,
            optimizer=child_agent.policy.optimizer,
            epochs=5,             # 離線訓練10輪
            lr=0.008780571292219902,               # 離線訓練使用稍大的學習率
            vf_coef=0.5,
            temperature=6.414784424905544,       # 軟目標的溫度
            alpha=0.5962410908066522,             # 軟目標的權重
            device=child_agent.device
        )
        ok = diff_report_before_after(policy_before, child_agent.policy, atol=1e-12)
        for h in hook_handles:
            h.remove()
        unfreeze_all(child_agent.policy)
        # 2.4 【關鍵】為子代策略設定我們特製的「精細版差異化優化器」
        # 注意：線上微調時，我們可能希望交叉通道的學習率高一些，所以這裡用了 lr_pure/10
        child_agent.policy.optimizer = create_refined_differential_optimizer(
            policy=child_agent.policy,
            lr_pure=child_agent.learning_rate, # 使用 PPO agent 的主學習率
            cross_lr_scale_factor= 0.2625007856712434 # 線上微調時，給予交叉通道 10% 的學習率
        )
        
        # 2.5 執行「全局離線蒸餾」作為熱身 (使用軟硬結合的損失函數)
        
        # 2.6 執行最終的「線上微調」
        print("\n--- 開始執行線上強化學習微調 ---")
        finetune_steps = 100000
        auto_difficulty_callback = AutoDifficultyCallback(
        env,None, eval_freq=10_000, reward_threshold=250, increase=0.05, verbose=1,shared_flags=False,cooldown_steps=0,hardseed_save_path="./logs/hard_seeds.json"
    )
        child_agent.learn(total_timesteps=finetune_steps, callback=[ auto_difficulty_callback,],progress_bar=True)

        evolved = child_agent # 將訓練完成的 agent 賦值給 evolved 變數
        
        # --- 步驟 3: 儲存訓練好的模型 (這部分保持不變) ---
        print("\n--- 正在儲存融合後的模型 ---")
        out_dir = Path("./models"); out_dir.mkdir(parents=True, exist_ok=True)
        dad_base = os.path.splitext(os.path.basename(dad_path))[0]
        mom_base = os.path.splitext(os.path.basename(mom_path))[0]
        tag = f"child_{dad_base}_x_{mom_base}_fused" # 加上 fused 標籤以區分
        zip_path = out_dir / f"{tag}.zip"
        evolved.save(str(zip_path))
        pkl_path = out_dir / f"{tag}_full.pkl"
        evolved.env = None # 移除 env 以便打包
        with open(pkl_path, "wb") as f:
            # 需要 import cloudpickle
            import cloudpickle
            cloudpickle.dump(evolved, f)

        print("✅ Saved:")
        print(f"   - SB3 zip:   {zip_path}")
        print(f"   - Full .pkl: {pkl_path}")
    elif args.find_value:
        study = optuna.create_study(
        direction="maximize",
        study_name="BipedalWalker_Fusion_HPO",
        # 您可以將 Optuna 的日誌儲存到資料庫中，以便後續分析
        storage="sqlite:///hpo_results.db", 
        load_if_exists=True
    )

        # 執行優化，例如進行 100 次完整的實驗
        study.optimize(objective, n_trials=100,n_jobs=1)

        # ----------------------------------------------------
        # 3. 輸出最佳結果
        # ----------------------------------------------------
        print("\n\n===== HPO 完成！=====")
        print("最佳 Trial 編號:", study.best_trial.number)
        print("最佳平均獎勵:", study.best_value)
        print("最佳超參數組合:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        print("\n--- 正在生成並儲存視覺化分析圖表 ---")
        
        # 創建一個專門存放圖表的資料夾
        output_dir = Path("./hpo_plots")
        output_dir.mkdir(exist_ok=True)
        
        # 檢查是否有已完成的 Trial
        if len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])) > 0:
            
            # --- A) 儲存 Slice Plot ---
            slice_fig = optuna.visualization.plot_slice(study)
            slice_path = output_dir / f"{study.study_name}_slice_plot.html"
            # 使用 .write_html() 儲存為網頁檔案
            slice_fig.write_html(str(slice_path))
            print(f"  - 已儲存 Slice Plot 至: {slice_path}")

            # --- B) 儲存參數重要性圖 ---
            try:
                importance_fig = optuna.visualization.plot_param_importances(study)
                importance_path = output_dir / f"{study.study_name}_importance_plot.html"
                importance_fig.write_html(str(importance_path))
                print(f"  - 已儲存 Importance Plot 至: {importance_path}")
            except Exception as e:
                print(f"\n  - 無法生成參數重要性圖: {e}")
            
            print("\n請用您的網頁瀏覽器開啟以上 .html 檔案來查看互動式圖表。")
        else:
            print("\n沒有已完成的 Trial，無法生成視覺化圖表。")
    
    else:
        parser.print_help()




