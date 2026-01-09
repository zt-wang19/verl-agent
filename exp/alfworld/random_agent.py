import json
import logging
import os
os.environ['ALFWORLD_DATA']='/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/data/alfworld'
import random
import time

# 添加uuid导入
import uuid
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import (
    alfworld_projection,
    build_alfworld_envs,
)
from agent_system.environments.prompts.alfworld import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
)

from collections import defaultdict

def build_val_env_manager(
    env_num: int, max_steps: int, seed: int = 1
) -> AlfWorldEnvironmentManager:
    """构建仅包含验证环境的AlfWorld管理器，参考make_envs的逻辑。"""

    group_n = 1
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
    eval_dataset = "eval_in_distribution"

    # 与make_envs保持一致的配置文件路径
    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        "../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )

    env_kwargs = {
        "eval_dataset": eval_dataset,
    }

    # 只构建验证环境（is_train=False），使用seed + 1000保持与make_envs一致
    val_envs = build_alfworld_envs(
        alf_config_path,
        seed + 1000,
        env_num,
        group_n,
        resources_per_worker=resources_per_worker,
        is_train=False,
        env_kwargs=env_kwargs,
    )

    # 构造最小配置供EnvironmentManager使用
    config = OmegaConf.create(
        {
            "env": {
                "env_name": "alfworld/AlfredThorEnv",
                "seed": seed,
                "rollout": {"n": 1},
                "resources_per_worker": resources_per_worker,
                "history_length": max_steps,
                "max_steps": max_steps,
                "alfworld": {"eval_dataset": eval_dataset},
            },
            "data": {
                "train_batch_size": 1,
                "val_batch_size": env_num,
            },
        }
    )

    projection_f = partial(alfworld_projection)
    return AlfWorldEnvironmentManager(val_envs, projection_f, config)


def build_train_env_manager(
    env_num: int, max_steps: int, seed: int = 1
) -> AlfWorldEnvironmentManager:
    """构建仅包含训练环境的AlfWorld管理器，便于脚本中按需调用。"""

    group_n = 1
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}

    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        "../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )

    env_kwargs = {
        "eval_dataset": "eval_in_distribution",
    }

    train_envs = build_alfworld_envs(
        alf_config_path,
        seed,
        env_num,
        group_n,
        resources_per_worker=resources_per_worker,
        is_train=True,
        env_kwargs=env_kwargs,
    )

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "alfworld/AlfredThorEnv",
                "seed": seed,
                "rollout": {"n": 1},
                "resources_per_worker": resources_per_worker,
                "history_length": max_steps,
                "max_steps": max_steps,
                "alfworld": {"eval_dataset": "eval_in_distribution"},
            },
            "data": {
                "train_batch_size": env_num,
                "val_batch_size": 1,
            },
        }
    )

    projection_f = partial(alfworld_projection)
    return AlfWorldEnvironmentManager(train_envs, projection_f, config)


class RandomAgent:
    def __init__(self):
        pass

    def get_action(self, prompt: str, admissible_commands):
        if admissible_commands:
            return random.choice(admissible_commands)
        else:
            return "None"


class ModelBasedAgent:
    """基于模型概率的采样Agent"""
    
    def __init__(self, model_path: str, device: str = "cuda", temperature: float = 1.0):
        """
        Args:
            model_path: 模型路径或HuggingFace模型名
            device: 运行设备
            temperature: 采样温度，用于调节概率分布的平滑程度
        """
        self.device = device
        self.temperature = temperature
        
        logging.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info("Model loaded successfully.")
    
    def compute_action_logprobs(self, prompt: str, actions: list) -> np.ndarray:
        """
        计算每个action的平均token log probability
        
        Args:
            prompt: 输入prompt
            actions: action列表
            
        Returns:
            每个action的平均log probability数组
        """
        logprobs = []
        
        with torch.no_grad():
            for action in actions:
                # 构建完整序列：prompt + action
                full_text = prompt + action
                prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
                
                # action的token从prompt结束位置开始
                action_start_idx = prompt_ids.shape[1]
                action_length = full_ids.shape[1] - action_start_idx
                
                if action_length <= 0:
                    # 如果action为空或者编码后没有额外token
                    logprobs.append(-float('inf'))
                    continue
                
                # 前向传播获取logits
                outputs = self.model(full_ids)
                logits = outputs.logits  # [1, seq_len, vocab_size]
                
                # 计算每个action token的log probability
                # logits[t] 预测的是 token[t+1]
                action_logprob = 0.0
                for i in range(action_start_idx, full_ids.shape[1]):
                    # logits at position i-1 predicts token at position i
                    token_id = full_ids[0, i]
                    log_probs = F.log_softmax(logits[0, i - 1], dim=-1)
                    action_logprob += log_probs[token_id].item()
                
                # 计算平均log probability
                avg_logprob = action_logprob / action_length
                logprobs.append(avg_logprob)
        
        return np.array(logprobs)
    
    def generate_actions(self, prompt: str, num_samples: int = 5, max_new_tokens: int = 50) -> list:
        """
        让模型生成多个action
        
        Args:
            prompt: 输入prompt
            num_samples: 生成的action数量
            max_new_tokens: 每个action的最大token数
            
        Returns:
            生成的action列表
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码生成的action（去掉prompt部分）
        prompt_length = inputs["input_ids"].shape[1]
        actions = []
        for output in outputs:
            action_ids = output[prompt_length:]
            action = self.tokenizer.decode(action_ids, skip_special_tokens=True).strip()
            # 只取第一行作为action（避免生成过长内容）
            action = action.split('\n')[0].strip()
            if action:
                actions.append(action)
        
        # 去重
        actions = list(set(actions))
        
        return actions if actions else ["look"]
    
    def get_action(self, prompt: str, admissible_commands: list, num_samples: int = 5) -> str:
        """
        根据模型概率获取action
        
        Args:
            prompt: 输入prompt
            admissible_commands: 可选action列表（可能为空）
            num_samples: 无admissible时生成的action数量
            
        Returns:
            选中的action
        """
        if admissible_commands:
            # 情况1: 有admissible actions，根据模型概率采样
            logprobs = self.compute_action_logprobs(prompt, admissible_commands)
            
            # 用temperature调节后转换为概率
            logprobs_scaled = logprobs / self.temperature
            # 防止数值溢出
            logprobs_scaled = logprobs_scaled - np.max(logprobs_scaled)
            probs = np.exp(logprobs_scaled)
            probs = probs / probs.sum()
            
            # 按概率采样
            action_idx = np.random.choice(len(admissible_commands), p=probs)
            return admissible_commands[action_idx]
        else:
            # 情况2: 没有admissible actions，生成多个action然后随机选择
            generated_actions = self.generate_actions(prompt, num_samples)
            return random.choice(generated_actions)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run agent on AlfWorld environment")
    parser.add_argument("--agent_type", type=str, default="random", choices=["random", "model"],
                        help="Agent type: 'random' for random sampling, 'model' for model-based sampling")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model (required if agent_type is 'model')")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for model-based agent")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate when no admissible actions")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Dataset split to use")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--max_concurrent_env_num", type=int, default=128,
                        help="Maximum concurrent environments")
    parser.add_argument("--target_env_num", type=int, default=640,
                        help="Target number of environments to process")
    args = parser.parse_args()

    # -------- Parameters ----------
    split = args.split
    max_steps = args.max_steps
    max_concurrent_env_num = args.max_concurrent_env_num
    target_env_num = args.target_env_num

    log_dir = f"data/exp/logs_alfworld"
    trajectory_dir = (
        f"data/exp/trajectories_alfworld"
    )
    # -------- logging ----------
    os.makedirs(log_dir, exist_ok=True)
    log_fp = os.path.join(
        log_dir, f"{args.agent_type}_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_fp, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # -------- Environment and agent setup ----------
    if args.agent_type == "model":
        if args.model_path is None:
            raise ValueError("--model_path is required when using model-based agent")
        agent = ModelBasedAgent(
            model_path=args.model_path,
            device=args.device,
            temperature=args.temperature
        )
        # 保存num_samples供后续使用
        agent_num_samples = args.num_samples
    else:
        agent = RandomAgent()
        agent_num_samples = 1  # RandomAgent不使用这个参数
    env_manager = (
        build_train_env_manager(env_num=max_concurrent_env_num, max_steps=max_steps)
        if split == "train"
        else build_val_env_manager(env_num=max_concurrent_env_num, max_steps=max_steps)
    )
    env_num = max_concurrent_env_num
    total_trajectories = []
    processed_envs = 0
    batch_idx = 0

    logging.info(f"\n========== Start {args.agent_type} agent interaction ==========")
    start_time = time.time()
    
    total_output_obs = []
    total_output_actions = []
    total_output_admissibles = []
    total_tasks = []

    while processed_envs < target_env_num:
        current_env_num = min(env_num, target_env_num - processed_envs)
        logging.info(
            f"\n------ Batch {batch_idx + 1}: running {current_env_num} envs "
            f"(processed {processed_envs}/{target_env_num}) ------"
        )

        # Trajectories storage for current batch
        trajectories = [[] for _ in range(env_num)]

        # 初始化额外变量，对齐vanilla_multi_turn_loop
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(env_num)], dtype=object)
        is_done = np.zeros(env_num, dtype=bool)
        episode_lengths = np.zeros(env_num, dtype=np.float32)
        episode_rewards = np.zeros(env_num, dtype=np.float32)
        total_infos = [[] for _ in range(env_num)]

        # 在reset后获取任务与初始观察
        obs, infos = env_manager.reset({})
        tasks = getattr(env_manager, "tasks", None)
        total_tasks.extend(tasks)

        batch_start_time = time.time()


        output_obs = defaultdict(list)
        output_actions = defaultdict(list)
        output_admissibles = defaultdict(list)
        
        # ======================= Main Loop =======================
        for _step in range(max_steps):
            logging.info(
                f"[Batch {batch_idx + 1}] Step {_step}; "
                f"Dones ({np.array(is_done).sum().item()}/{env_num})"
            )

            active_masks = np.logical_not(is_done)

            # --- Prepare prompts and actions ---
            admissible_commands_list = env_manager.envs.get_admissible_commands
            memory_contexts_before, _ = env_manager.memory.fetch(
                history_length=max_steps,
                obs_key="text_obs",
                action_key="action",
            )

            prompts = [None] * env_num
            actions = []
            
            admissibles = {}
            
            for i in range(env_num):
                if active_masks[i]:
                    admissible = (
                        admissible_commands_list[i] if admissible_commands_list else []
                    )
                    reformatted_admissible = (
                        "\n ".join(f"'{s}'" for s in admissible if s != "help")
                        or "None"
                    )
                    admissibles[i]=admissible
                    history_text_before = memory_contexts_before[i]
                    step_count_before = len(env_manager.memory[i])
                    current_step = step_count_before + 1
                    current_observation = obs["anchor"][i]

                    if step_count_before == 0:
                        prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                            current_observation=current_observation,
                            admissible_actions=reformatted_admissible,
                        )
                    else:
                        prompt = ALFWORLD_TEMPLATE.format(
                            task_description=tasks[i],
                            step_count=step_count_before,
                            history_length=2,
                            action_history=history_text_before,
                            current_step=current_step,
                            current_observation=current_observation,
                            admissible_actions=reformatted_admissible,
                        )

                    if isinstance(agent, ModelBasedAgent):
                        action = agent.get_action(prompt, admissible, num_samples=agent_num_samples)
                    else:
                        action = agent.get_action(prompt, admissible)
                    prompts[i] = prompt
                    actions.append(action)
                else:
                    prompts[i] = None
                    actions.append("None")
            
            for i in range(env_num):
                if active_masks[i]:
                    output_obs[i].append(obs["anchor"][i])
                    output_actions[i].append(actions[i])
                    output_admissibles[i].append(admissibles[i])
            
            # --- Environment stepping ---
            next_obs, rewards, dones, next_infos = env_manager.step(actions)

            # 更新rewards和lengths
            rewards_np = np.array(rewards)
            dones_np = np.array(dones)
            episode_rewards[active_masks] += rewards_np[active_masks]
            episode_lengths[active_masks] += 1

            # 提取历史（SimpleMemory格式，step后包含最新交互）
            memory_contexts_after, _ = env_manager.memory.fetch(
                history_length=max_steps,
                obs_key="text_obs",
                action_key="action",
            )

            # 在循环中，为active环境记录轨迹
            for i in range(env_num):
                if active_masks[i]:
                    admissible = admissibles[i]
                    history_text = memory_contexts_after[i]
                    step_count_after = len(env_manager.memory[i])

                    trajectories[i].append(
                        {
                            "traj_uid": traj_uid[i],
                            "prompt": prompts[i],
                            "history": history_text,
                            "observation": obs["anchor"][i],
                            "action": actions[i],
                            "reward": float(rewards[i]),
                            "next_observation": next_obs["anchor"][i],
                            "done": bool(dones[i]),
                            "active_masks": True,
                            "is_action_valid": bool(
                                next_infos[i].get("is_action_valid", True)
                            ),
                            "admissible_actions": admissible,
                            "step_index": step_count_after,
                        }
                    )
                    total_infos[i].append(next_infos[i])

            # 更新done并刷新观测
            is_done = np.logical_or(is_done, dones_np)
            obs = next_obs

            if is_done.all():
                logging.info(
                    f"[Batch {batch_idx + 1}] All environments finished early!"
                )
                break

        # 计算success，如果envs有success_evaluator
        if hasattr(env_manager, "success_evaluator"):
            success = env_manager.success_evaluator(
                total_infos=total_infos,
                total_batch_list=trajectories,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
            )
            logging.info(f"[Batch {batch_idx + 1}] Success metrics: {success}")
        else:
            logging.info(f"[Batch {batch_idx + 1}] No success_evaluator available.")

        total_output_obs.extend(output_obs.values())
        total_output_actions.extend(output_actions.values())
        total_output_admissibles.extend(output_admissibles.values())
        total_trajectories.extend(trajectories[:current_env_num])
        processed_envs += current_env_num
        batch_idx += 1
        logging.info(
            f"Batch {batch_idx} finished in {time.time() - batch_start_time:.2f}s; "
            f"total processed envs: {processed_envs}/{target_env_num}"
        )

    if hasattr(env_manager, "close"):
        env_manager.close()

    # -------- Save trajectories --------
    os.makedirs(trajectory_dir, exist_ok=True)
    output_json = [{
        'task': total_tasks[i],
        'obs': total_output_obs[i],
        'actions': total_output_actions[i],
        'admissibles': total_output_admissibles[i]
    } for i in range(len(total_output_obs))]
    trajectory_fp = os.path.join(
        trajectory_dir,
        f"{args.agent_type}_trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(trajectory_fp, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)

    logging.info(f"Trajectories saved to {trajectory_fp}")
    logging.info(f"Time elapsed: {time.time() - start_time:.2f}s\n")
