# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gymnasium as gym
import numpy as np
import ray
import random
from tqdm import tqdm


class ScienceWorldWorker:
    """
    Ray remote actor for ScienceWorld environment.
    Each actor holds one environment instance.
    Supports random task selection weighted by variation counts.
    """
    
    def __init__(self, task_name, seed, simplification_str="easy", env_step_limit=100, is_train=True):
        # print(f"[DEBUG] ScienceWorldWorker.__init__ called for seed={seed}")
        self.task_name = task_name  # Can be "all" for random task selection
        self.seed = seed
        self.simplification_str = simplification_str
        self.env_step_limit = env_step_limit
        self.is_train = is_train
        self.env = None
        self.current_task_name = None
        self.current_variation_idx = 0
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        self._init_env()
        # print(f"[DEBUG] ScienceWorldWorker.__init__ finished for seed={seed}")
    
    def _init_env(self):
        """Initialize the ScienceWorld environment and build task distribution"""
        # print(f"[DEBUG] _init_env start for seed={self.seed}")
        # Create env without loading a specific task first
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv(envStepLimit=self.env_step_limit)
        
        # Get all task names
        self.all_task_names = self.env.get_task_names()
        # print(f"[DEBUG] Found {len(self.all_task_names)} tasks for seed={self.seed}")
        
        # Build variation counts for each task (for weighted random selection)
        self.task_variation_counts = {}
        self.task_variations = {}
        
        for task in tqdm(self.all_task_names, desc=f"Loading tasks (seed={self.seed})", disable=True):
            # print(f"[DEBUG] Loading task {task} ({i+1}/{len(self.all_task_names)}) for seed={self.seed}")
            self.env.load(task, 0, self.simplification_str)
            if self.is_train:
                variations = self.env.get_variations_train()
            else:
                variations = self.env.get_variations_test()
            self.task_variations[task] = variations
            self.task_variation_counts[task] = len(variations)
        
        # print(f"[DEBUG] Finished loading all tasks for seed={self.seed}")
        
        # Build probability distribution based on variation counts
        total_variations = sum(self.task_variation_counts.values())
        self.task_probs = {
            task: count / total_variations 
            for task, count in self.task_variation_counts.items()
        }
        self.task_names_list = list(self.task_probs.keys())
        self.task_probs_list = [self.task_probs[t] for t in self.task_names_list]
        
        # Load initial task
        # print(f"[DEBUG] Loading initial random task for seed={self.seed}")
        self._load_random_task_and_variation()
        # print(f"[DEBUG] _init_env done for seed={self.seed}")
    
    def _load_random_task_and_variation(self):
        """Randomly select a task (weighted by variation count) and a variation"""
        if self.task_name == "all":
            # Random task selection weighted by variation counts
            self.current_task_name = np.random.choice(
                self.task_names_list, 
                p=self.task_probs_list
            )
        else:
            # Use fixed task name
            self.current_task_name = self.task_name
        
        # Get variations for current task
        variations = self.task_variations[self.current_task_name]
        
        # Random variation from train/test split
        self.current_variation_idx = random.choice(variations)
        
        # Load the task with selected variation
        self.env.load(
            self.current_task_name,
            self.current_variation_idx,
            self.simplification_str
        )
    
    def step(self, action):
        """Execute a step in the environment"""
        # print(f"[DEBUG] Worker step start. seed={self.seed}, action={action}")
        observation, reward, done, info = self.env.step(action)
        
        # Add additional info
        info['task_name'] = self.current_task_name
        info['variation_idx'] = self.current_variation_idx
        info['won'] = info.get('score', 0) >= 100  # Task is completed when score reaches 100
        info['goal_progress'] = self.env.get_goal_progress()
        
        # print(f"[DEBUG] Worker step done. seed={self.seed}, done={done}")
        return observation, reward, done, info
    
    def reset(self):
        """Reset the environment with a new random task and variation"""
        # print(f"[DEBUG] Worker reset start. seed={self.seed}")
        # Select new random task and variation
        self._load_random_task_and_variation()
        
        observation, info = self.env.reset()
        
        # Add additional info
        info['task_name'] = self.current_task_name
        info['variation_idx'] = self.current_variation_idx
        info['task_description'] = self.env.get_task_description()
        info['valid_actions'] = self.env.get_valid_action_object_combinations()
        info['possible_actions'] = self.env.get_possible_actions()
        
        # print(f"[DEBUG] Worker reset done. seed={self.seed}")
        return observation, info
    
    def get_valid_actions(self):
        """Get valid actions for the current state"""
        return self.env.get_valid_action_object_combinations()
    
    def get_possible_actions(self):
        """Get possible action templates"""
        return self.env.get_possible_actions()
    
    def get_task_description(self):
        """Get the current task description"""
        return self.env.get_task_description()
    
    def look(self):
        """Look around (free action)"""
        return self.env.look()
    
    def inventory(self):
        """Check inventory (free action)"""
        return self.env.inventory()
    
    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()
    
    def get_task_distribution(self):
        """Return the task probability distribution for debugging"""
        return self.task_probs


class ScienceWorldEnvs(gym.Env):
    """
    Vectorized ScienceWorld environment using Ray for parallelization.
    Supports random task selection weighted by variation counts.
    """
    
    def __init__(
        self,
        task_name,
        seed,
        env_num,
        group_n,
        resources_per_worker,
        is_train=True,
        simplification_str="easy",
        env_step_limit=100
    ):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.task_name = task_name  # Can be "all" for random task selection
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        self.simplification_str = simplification_str
        self.env_step_limit = env_step_limit
        
        # Create Ray remote actors
        print(f"[DEBUG] ScienceWorldEnvs: Creating {self.num_processes} workers...")
        env_worker = ray.remote(**resources_per_worker)(ScienceWorldWorker)
        self.workers = []
        for i in range(self.num_processes):
            print(f"[DEBUG] ScienceWorldEnvs: Spawning worker {i+1}/{self.num_processes}")
            worker = env_worker.remote(
                task_name=task_name,
                seed=seed + i,  # Each worker gets unique seed
                simplification_str=simplification_str,
                env_step_limit=env_step_limit,
                is_train=is_train  # Pass is_train to select train/test split
            )
            self.workers.append(worker)
        print(f"[DEBUG] ScienceWorldEnvs: All workers spawned.")
        
        # Cache for valid actions per environment
        self.prev_valid_actions = [None for _ in range(self.num_processes)]
        self.prev_task_descriptions = [None for _ in range(self.num_processes)]
    
    def step(self, actions):
        """
        Execute actions in all environments.
        
        Args:
            actions: List of action strings
            
        Returns:
            text_obs_list: List of observation strings
            rewards_list: List of rewards
            dones_list: List of done flags
            info_list: List of info dicts
        """
        # print(f"[DEBUG] ScienceWorldEnvs: step called with {len(actions)} actions")
        assert len(actions) == self.num_processes, \
            f"Number of actions ({len(actions)}) must equal number of processes ({self.num_processes})"
        
        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)
        
        # Collect results
        # print(f"[DEBUG] ScienceWorldEnvs: waiting for step results...")
        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []
        
        results = ray.get(futures)
        # print(f"[DEBUG] ScienceWorldEnvs: step results received.")
        for i, (obs, reward, done, info) in enumerate(results):
            text_obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            info_list.append(info)
            
            # Update cached valid actions
            self.prev_valid_actions[i] = info.get('valid', [])
        
        return text_obs_list, rewards_list, dones_list, info_list
    
    def reset(self):
        """
        Reset all environments.
        
        Returns:
            text_obs_list: List of initial observation strings
            info_list: List of info dicts
        """
        print(f"[DEBUG] ScienceWorldEnvs: reset called")
        text_obs_list = []
        info_list = []
        
        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)
        
        # Collect results
        print(f"[DEBUG] ScienceWorldEnvs: waiting for reset results...")
        results = ray.get(futures)
        print(f"[DEBUG] ScienceWorldEnvs: reset results received.")
        for i, (obs, info) in enumerate(results):
            text_obs_list.append(obs)
            info_list.append(info)
            
            # Update cached valid actions and task descriptions
            self.prev_valid_actions[i] = info.get('valid_actions', [])
            self.prev_task_descriptions[i] = info.get('task_description', '')
        
        return text_obs_list, info_list
    
    @property
    def get_valid_actions(self):
        """Get cached valid actions for all environments"""
        return self.prev_valid_actions
    
    @property
    def get_task_descriptions(self):
        """Get cached task descriptions for all environments"""
        return self.prev_task_descriptions
    
    def close(self):
        """Close all workers"""
        for worker in self.workers:
            ray.kill(worker)


def build_scienceworld_envs(
    task_name,
    seed,
    env_num,
    group_n,
    resources_per_worker,
    is_train=True,
    simplification_str="easy",
    env_step_limit=100
):
    """
    Build ScienceWorld vectorized environments.
    
    Args:
        task_name: Name of the task (e.g., "boil", "melt", "freeze"), or "all" for 
                   random task selection weighted by variation counts.
        seed: Random seed
        env_num: Number of environments
        group_n: Group size for rollout
        resources_per_worker: Ray resource configuration per worker
        is_train: Whether this is for training. If True, uses train split variations;
                  if False, uses test split variations.
        simplification_str: Simplification string (e.g., "easy")
        env_step_limit: Maximum steps per episode
        
    Returns:
        ScienceWorldEnvs instance
        
    Note:
        When task_name="all", each reset() will:
        1. Randomly select a task with probability proportional to its variation count
        2. Randomly select a variation from train/test split based on is_train
    """
    return ScienceWorldEnvs(
        task_name=task_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        simplification_str=simplification_str,
        env_step_limit=env_step_limit
    )

