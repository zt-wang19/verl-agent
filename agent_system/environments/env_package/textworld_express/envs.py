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

import gymnasium as gym
import numpy as np
import ray
import random


class TextWorldExpressWorker:
    """
    Ray remote actor for TextWorldExpress environment.
    Each actor holds one environment instance.
    Supports multiple game types: cookingworld, coin, twc, mapreader, arithmetic, sorting, simonsays, peckingorder.
    """
    
    def __init__(
        self, 
        game_name, 
        seed, 
        env_step_limit=50, 
        is_train=True,
        game_params=""
    ):
        self.game_name = game_name  # Can be "all" for random game selection
        self.seed = seed
        self.env_step_limit = env_step_limit
        self.is_train = is_train
        self.game_params = game_params
        self.env = None
        self.current_game_name = None
        self.current_seed = None
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # All available game names
        self.all_game_names = [
            'cookingworld', 'coin', 'twc', 'mapreader', 
            'arithmetic', 'sorting', 'simonsays', 'peckingorder'
        ]
        
        self._init_env()
    
    def _init_env(self):
        """Initialize the TextWorldExpress environment"""
        from textworld_express import TextWorldExpressEnv
        self.env = TextWorldExpressEnv(envStepLimit=self.env_step_limit)
        
        # Set initial game
        self._load_random_game()
    
    def _load_random_game(self):
        """Randomly select a game and seed"""
        if self.game_name == "all":
            # Random game selection
            self.current_game_name = random.choice(self.all_game_names)
        else:
            self.current_game_name = self.game_name
        
        # Load the game
        self.env.load(self.current_game_name, self.game_params)
        
        # Get fold based on train/test
        game_fold = "train" if self.is_train else "test"
        
        # Get a random seed for this fold
        if self.is_train:
            self.current_seed = self.env.getRandomSeedTrain()
        else:
            self.current_seed = self.env.getRandomSeedTest()
    
    def step(self, action):
        """Execute a step in the environment"""
        observation, reward, done, info = self.env.step(action)
        
        # Add additional info
        info['game_name'] = self.current_game_name
        info['current_seed'] = self.current_seed
        info['won'] = info.get('tasksuccess', False)
        info['valid'] = info.get('validActions', [])
        
        return observation, reward, done, info
    
    def reset(self):
        """Reset the environment with a new random game and seed"""
        # Select new random game
        self._load_random_game()
        
        # Reset with the selected seed
        game_fold = "train" if self.is_train else "test"
        observation, info = self.env.reset(
            seed=self.current_seed, 
            gameFold=game_fold,
            generateGoldPath=False
        )
        
        # Add additional info
        info['game_name'] = self.current_game_name
        info['current_seed'] = self.current_seed
        info['task_description'] = self.env.getTaskDescription()
        info['valid'] = info.get('validActions', [])
        
        return observation, info
    
    def get_valid_actions(self):
        """Get valid actions for the current state"""
        # Get the last info from run history
        if len(self.env.runHistory) > 0:
            return self.env.runHistory[-1].get('validActions', [])
        return []
    
    def get_task_description(self):
        """Get the current task description"""
        return self.env.getTaskDescription()
    
    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()


class TextWorldExpressEnvs(gym.Env):
    """
    Vectorized TextWorldExpress environment using Ray for parallelization.
    Supports multiple game types: cookingworld, coin, twc, mapreader, arithmetic, sorting, simonsays, peckingorder.
    """
    
    def __init__(
        self,
        game_name,
        seed,
        env_num,
        group_n,
        resources_per_worker,
        is_train=True,
        env_step_limit=50,
        game_params=""
    ):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.game_name = game_name  # Can be "all" for random game selection
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        self.env_step_limit = env_step_limit
        self.game_params = game_params
        
        # Create Ray remote actors
        env_worker = ray.remote(**resources_per_worker)(TextWorldExpressWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(
                game_name=game_name,
                seed=seed + i,  # Each worker gets unique seed
                env_step_limit=env_step_limit,
                is_train=is_train,
                game_params=game_params
            )
            self.workers.append(worker)
        
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
        assert len(actions) == self.num_processes, \
            f"Number of actions ({len(actions)}) must equal number of processes ({self.num_processes})"
        
        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)
        
        # Collect results
        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []
        
        results = ray.get(futures)
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
        text_obs_list = []
        info_list = []
        
        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            text_obs_list.append(obs)
            info_list.append(info)
            
            # Update cached valid actions and task descriptions
            self.prev_valid_actions[i] = info.get('valid', [])
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


def build_textworld_express_envs(
    game_name,
    seed,
    env_num,
    group_n,
    resources_per_worker,
    is_train=True,
    env_step_limit=50,
    game_params=""
):
    """
    Build TextWorldExpress vectorized environments.
    
    Args:
        game_name: Name of the game (e.g., "cookingworld", "coin", "twc"), or "all" for 
                   random game selection.
        seed: Random seed
        env_num: Number of environments
        group_n: Group size for rollout
        resources_per_worker: Ray resource configuration per worker
        is_train: Whether this is for training. If True, uses train split;
                  if False, uses test split.
        env_step_limit: Maximum steps per episode
        game_params: Game-specific parameters (e.g., "numLocations=5, includeDoors=1")
        
    Returns:
        TextWorldExpressEnvs instance
        
    Note:
        Available game names: cookingworld, coin, twc, mapreader, arithmetic, sorting, simonsays, peckingorder
        When game_name="all", each reset() will randomly select a game.
    """
    return TextWorldExpressEnvs(
        game_name=game_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_step_limit=env_step_limit,
        game_params=game_params
    )

