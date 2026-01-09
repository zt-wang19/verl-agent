import json
import logging
import os
import random
import time
import uuid
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
from omegaconf import OmegaConf

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.env_package.webshop import (
    build_webshop_envs,
    webshop_projection,
)


def _resolve_webshop_paths(use_small: bool) -> Tuple[str, str]:
    """根据配置返回 WebShop 所需的数据文件路径。"""
    data_root = os.path.join(
        os.path.dirname(__file__),
        "../../agent_system/environments/env_package/webshop/webshop/data",
    )

    if use_small:
        items_filename = "items_shuffle_1000.json"
        attrs_filename = "items_ins_v2_1000.json"
    else:
        items_filename = "items_shuffle.json"
        attrs_filename = "items_ins_v2.json"

    file_path = os.path.abspath(os.path.join(data_root, items_filename))
    attr_path = os.path.abspath(os.path.join(data_root, attrs_filename))
    return file_path, attr_path


def build_train_env_manager(
    env_num: int,
    max_steps: int,
    seed: int = 0,
    use_small: bool = True,
    human_goals: bool = False,
) -> WebshopEnvironmentManager:
    """构建仅包含训练环境的 WebShop 环境管理器。"""
    group_n = 1
    resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}

    file_path, attr_path = _resolve_webshop_paths(use_small)
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": human_goals,
        "file_path": file_path,
        "attr_path": attr_path,
    }

    envs = build_webshop_envs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=True,
        env_kwargs=env_kwargs,
    )

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "webshop/WebAgentTextEnv",
                "seed": seed,
                "rollout": {"n": group_n},
                "resources_per_worker": resources_per_worker,
                "history_length": 2,
                "max_steps": max_steps,
                "webshop": {
                    "use_small": use_small,
                    "human_goals": human_goals,
                },
            },
            "data": {
                "train_batch_size": env_num,
                "val_batch_size": 1,
            },
        }
    )

    projection_f = partial(webshop_projection)
    return WebshopEnvironmentManager(envs, projection_f, config)


def build_val_env_manager(
    env_num: int,
    max_steps: int,
    seed: int = 0,
    use_small: bool = True,
    human_goals: bool = False,
) -> WebshopEnvironmentManager:
    """构建仅包含验证环境的 WebShop 环境管理器。"""
    group_n = 1
    resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}

    file_path, attr_path = _resolve_webshop_paths(use_small)
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": human_goals,
        "file_path": file_path,
        "attr_path": attr_path,
    }

    envs = build_webshop_envs(
        seed=seed + 1000,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=False,
        env_kwargs=env_kwargs,
    )

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "webshop/WebAgentTextEnv",
                "seed": seed,
                "rollout": {"n": group_n},
                "resources_per_worker": resources_per_worker,
                "history_length": 2,
                "max_steps": max_steps,
                "webshop": {
                    "use_small": use_small,
                    "human_goals": human_goals,
                },
            },
            "data": {
                "train_batch_size": 1,
                "val_batch_size": env_num,
            },
        }
    )

    projection_f = partial(webshop_projection)
    return WebshopEnvironmentManager(envs, projection_f, config)


class RandomWebshopAgent:
    """一个极简基线代理：从可行动作中随机采样。"""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def _sample_query(self, task_description: str) -> str:
        """根据任务描述随机挑选一个词作为搜索关键字。"""
        tokens = [
            token.strip(".,:;!?")
            for token in task_description.lower().split()
            if len(token.strip(".,:;!?")) >= 3
        ]
        if tokens:
            return self._rng.choice(tokens)
        return "gift"

    def get_action(
        self,
        available_actions: List[str],
        task_description: str,
    ) -> str:
        if not available_actions:
            query = self._sample_query(task_description)
            return f"search[{query}]"

        action = self._rng.choice(available_actions)
        if action.startswith("search["):
            query = self._sample_query(task_description)
            return f"search[{query}]"
        return action


def _format_available_actions(
    env_manager: WebshopEnvironmentManager, infos: List[Dict]
) -> List[List[str]]:
    formatted = []
    for info in infos:
        if not isinstance(info, dict):
            formatted.append([])
            continue

        avail = info.get("available_actions")
        if avail is None:
            formatted.append([])
            continue

        try:
            formatted.append(env_manager.format_avail_actions(avail))
        except Exception as exc:  # noqa: BLE001
            logging.warning("格式化可行动作失败: %s", exc)
            formatted.append([])
    return formatted


if __name__ == "__main__":
    # -------- 参数设置 ----------
    split = "train"  # "train" 或 "eval"
    max_steps = 15
    max_concurrent_env_num = 128 if split == "train" else 128
    target_env_num = 32 * 1100 // 15
    env_num = max_concurrent_env_num
    seed = 0
    use_small = True
    human_goals = False

    log_dir = f"data/exp/logs_webshop"
    trajectory_dir = (
        f"data/exp/trajectories_webshop"
    )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)

    log_fp = os.path.join(
        log_dir, f"random_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_fp, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info("WebShop 随机代理启动，split=%s, env_num=%d", split, env_num)

    # -------- 构建环境 ----------
    if split == "train":
        env_manager = build_train_env_manager(
            env_num=max_concurrent_env_num,
            max_steps=max_steps,
            seed=seed,
            use_small=use_small,
            human_goals=human_goals,
        )
    else:
        env_manager = build_val_env_manager(
            env_num=max_concurrent_env_num,
            max_steps=max_steps,
            seed=seed,
            use_small=use_small,
            human_goals=human_goals,
        )

    agent = RandomWebshopAgent(seed=seed)

    total_trajectories: List[List[Dict]] = []
    processed_envs = 0
    batch_idx = 0

    total_output_obs = []
    total_output_actions = []
    total_output_admissibles = []
    total_tasks = []

    logging.info("\n========== 开始随机交互 ==========")
    start_time = time.time()

    while processed_envs < target_env_num:
        current_env_num = min(env_num, target_env_num - processed_envs)
        print("DEBUG current_env_num:", current_env_num, type(current_env_num))
        logging.info(
            "\n------ Batch %d: running %d envs (processed %d/%d) ------",
            batch_idx + 1,
            current_env_num,
            processed_envs,
            target_env_num,
        )
        batch_start_time = time.time()

        trajectories: List[List[Dict]] = [[] for _ in range(env_num)]
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(env_num)], dtype=object)
        is_done = np.zeros(env_num, dtype=bool)
        if current_env_num < env_num:
            is_done[current_env_num:] = True
        episode_lengths = np.zeros(env_num, dtype=np.float32)
        episode_rewards = np.zeros(env_num, dtype=np.float32)
        total_infos: List[List[Dict]] = [[] for _ in range(env_num)]

        obs, infos = env_manager.reset({})
        tasks = getattr(env_manager, "tasks", [""] * env_num)
        total_tasks.extend(tasks[:current_env_num])
        available_actions_list = _format_available_actions(env_manager, infos)

        output_obs = defaultdict(list)
        output_actions = defaultdict(list)
        output_admissibles = defaultdict(list)

        for step_idx in range(max_steps):
            logging.info(
                "[Batch %d] Step %d; Dones (%d/%d)",
                batch_idx + 1,
                step_idx,
                int(np.array(is_done).sum().item()),
                env_num,
            )

            active_masks = np.logical_not(is_done)
            prompts = obs.get("text") or [None] * env_num

            actions: list[str] = []
            for env_idx in range(env_num):
                if active_masks[env_idx]:
                    task_desc = tasks[env_idx] if tasks else ""
                    action = agent.get_action(
                        available_actions_list[env_idx],
                        task_desc,
                    )
                    actions.append(action)
                else:
                    actions.append("search[noop]")

            for env_idx in range(env_num):
                if active_masks[env_idx]:
                    output_obs[env_idx].append(obs["anchor"][env_idx])
                    output_actions[env_idx].append(actions[env_idx])
                    output_admissibles[env_idx].append(available_actions_list[env_idx])

            next_obs, rewards, dones, next_infos = env_manager.step(actions)

            rewards_np = np.array(rewards)
            dones_np = np.array(dones)
            episode_rewards[active_masks] += rewards_np[active_masks]
            episode_lengths[active_masks] += 1

            memory_contexts_after, _ = env_manager.memory.fetch(
                history_length=max_steps,
                obs_key="text_obs",
                action_key="action",
            )

            for env_idx in range(env_num):
                if not active_masks[env_idx]:
                    continue

                trajectories[env_idx].append(
                    {
                        "traj_uid": traj_uid[env_idx],
                        "prompt": prompts[env_idx],
                        "history": memory_contexts_after[env_idx],
                        "observation": obs["anchor"][env_idx],
                        "action": actions[env_idx],
                        "reward": float(rewards_np[env_idx]),
                        "next_observation": next_obs["anchor"][env_idx],
                        "done": bool(dones_np[env_idx]),
                        "active_masks": True,
                        "is_action_valid": bool(
                            next_infos[env_idx].get("is_action_valid", True)
                        ),
                        "available_actions": available_actions_list[env_idx],
                        "step_index": len(env_manager.memory[env_idx]),
                    }
                )
                total_infos[env_idx].append(next_infos[env_idx])

            is_done = np.logical_or(is_done, dones_np)
            obs = next_obs
            infos = next_infos
            available_actions_list = _format_available_actions(env_manager, infos)

            if is_done.all():
                logging.info("[Batch %d] 所有环境提前完成。", batch_idx + 1)
                break

        if hasattr(env_manager, "success_evaluator"):
            success = env_manager.success_evaluator(
                total_infos=total_infos[:current_env_num],
                total_batch_list=trajectories[:current_env_num],
                episode_rewards=episode_rewards[:current_env_num],
                episode_lengths=episode_lengths[:current_env_num],
            )
            logging.info("[Batch %d] 成功统计: %s", batch_idx + 1, success)
        else:
            logging.info(
                "[Batch %d] 当前环境管理器未提供 success_evaluator。", batch_idx + 1
            )

        total_output_obs.extend(output_obs.values())
        total_output_actions.extend(output_actions.values())
        total_output_admissibles.extend(output_admissibles.values())
        total_trajectories.extend(trajectories[:current_env_num])
        processed_envs += current_env_num
        batch_idx += 1
        logging.info(
            "Batch %d 完成，耗时 %.2fs; 已处理 %d/%d 环境",
            batch_idx,
            time.time() - batch_start_time,
            processed_envs,
            target_env_num,
        )

    output_json = [
        {
            "task": total_tasks[i],
            "obs": total_output_obs[i],
            "actions": total_output_actions[i],
            "admissibles": total_output_admissibles[i],
        }
        for i in range(len(total_output_obs))
    ]
    trajectory_fp = os.path.join(
        trajectory_dir,
        f"random_trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(trajectory_fp, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)

    logging.info("轨迹已保存至 %s", trajectory_fp)
    logging.info("总耗时: %.2fs", time.time() - start_time)

    if hasattr(env_manager, "close"):
        env_manager.close()

    if ray.is_initialized():
        ray.shutdown()
