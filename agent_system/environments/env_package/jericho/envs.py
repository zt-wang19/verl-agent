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
from typing import List, Optional, Dict, Any


# List of supported Jericho games (from jericho.defines)
SUPPORTED_GAMES = [
    '905', 'acorncourt', 'adventureland', 'advent', 'afflicted', 'anchor',
    'awaken', 'balances', 'ballyhoo', 'curses', 'cutthroat', 'deephome',
    'detective', 'dragon', 'enchanter', 'enter', 'gold', 'hhgg', 'hollywood',
    'huntdark', 'infidel', 'inhumane', 'jewel', 'karn', 'library', 'loose',
    'lostpig', 'ludicorp', 'lurking', 'moonlit', 'murdac', 'night', 'omniquest',
    'partyfoul', 'pentari', 'planetfall', 'plundered', 'reverb', 'seastalker',
    'sherlock', 'snacktime', 'sorcerer', 'spellbrkr', 'spirit',
    'temple', 'theatre', 'trinity', 'tryst205', 'weapon', 'wishbringer', 'yomomma',
    'zenon', 'zork1', 'zork2', 'zork3', 'ztuu'
]

# Train/Test split by games (roughly 80/20 split)
# Train games: 43 games
TRAIN_GAMES = [
    '905', 'acorncourt', 'adventureland', 'advent', 'afflicted', 
    'awaken', 'balances', 'ballyhoo', 'curses', 'cutthroat', 'deephome',
    'detective', 'dragon', 'enchanter', 'enter', 'gold', 'hhgg', 'hollywood',
    'huntdark', 'infidel', 'inhumane', 'jewel', 'karn', 'library', 
    'lostpig', 'ludicorp', 'lurking', 'moonlit', 'murdac', 'night', 'omniquest',
    'partyfoul', 'pentari', 'planetfall', 'plundered', 'reverb', 
    'sherlock', 'snacktime', 'sorcerer', 'spellbrkr', 'spirit',
    'temple', 'theatre'
]

# Test games: 12 games (held out for evaluation)
TEST_GAMES = [
    'anchor', 'loose', 'seastalker', 'trinity', 'tryst205', 
    'weapon', 'wishbringer', 'yomomma', 'zenon', 'zork1', 'zork2', 'zork3', 'ztuu'
]


class JerichoWorker:
    """
    Ray remote actor for Jericho environment.
    Each actor holds one environment instance.
    """
    
    def __init__(
        self, 
        game_name: str,
        rom_path: str,
        seed: int, 
        env_step_limit: int = 100,
        is_train: bool = True
    ):
        self.game_name = game_name
        self.rom_path = rom_path
        self.seed = seed
        self.env_step_limit = env_step_limit
        self.is_train = is_train
        self.env = None
        self.current_step = 0
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        self._init_env()
    
    def _init_env(self):
        """Initialize the Jericho environment"""
        print(f"[DEBUG] JerichoWorker._init_env: Starting for game '{self.game_name}', seed={self.seed}")
        from jericho import FrotzEnv
        
        # Determine ROM path
        if os.path.isfile(self.rom_path):
            story_file = self.rom_path
        else:
            # Try to find the ROM file in standard locations
            possible_paths = [
                self.rom_path,
                f"{self.rom_path}.z5",
                f"{self.rom_path}.z8",
                f"{self.rom_path}.z3",
                os.path.join(os.path.dirname(__file__), 'roms', f"{self.game_name}.z5"),
                os.path.join(os.path.dirname(__file__), 'roms', f"{self.game_name}.z8"),
            ]
            story_file = None
            for path in possible_paths:
                if os.path.isfile(path):
                    story_file = path
                    break
            
            if story_file is None:
                raise FileNotFoundError(
                    f"Could not find ROM file for game '{self.game_name}'. "
                    f"Tried paths: {possible_paths}"
                )
        
        print(f"[DEBUG] JerichoWorker._init_env: Loading ROM from '{story_file}'")
        self.env = FrotzEnv(story_file, seed=self.seed)
        print(f"[DEBUG] JerichoWorker._init_env: FrotzEnv created successfully for '{self.game_name}'")
    
    def step(self, action: str):
        """Execute a step in the environment"""
        import time
        t0 = time.time()
        
        # Execute action
        t_step_start = time.time()
        try:
            observation, reward, done, info = self.env.step(action)
        except Exception as e:
            print(f"[ERROR] JerichoWorker.step: env.step failed with error: {e}")
            raise
        t_step_time = time.time() - t_step_start
        
        self.current_step += 1
        
        # Check if we've exceeded the step limit
        if self.current_step >= self.env_step_limit:
            done = True
        
        # Add additional info
        info['game_name'] = self.game_name
        info['current_step'] = self.current_step
        info['max_score'] = self.env.get_max_score()
        info['won'] = self.env.victory()
        info['game_over'] = self.env.game_over()
        
        # Get valid actions for next step (if game supports it) - THIS CAN BE SLOW!
        t_valid_start = time.time()
        try:
            info['valid'] = self.env.get_valid_actions()
        except Exception as e:
            info['valid'] = []
        t_valid_time = time.time() - t_valid_start
        
        # Get inventory
        t_inv_start = time.time()
        try:
            inventory = self.env.get_inventory()
            info['inv'] = ', '.join([obj.name for obj in inventory]) if inventory else 'empty'
        except Exception as e:
            info['inv'] = 'unknown'
        t_inv_time = time.time() - t_inv_start
        
        t_total = time.time() - t0
        if t_total > 1.0:  # Only print if slow (>1s)
            print(f"[TIMING] JerichoWorker.step SLOW: game='{self.game_name}', total={t_total:.2f}s "
                  f"(step={t_step_time:.2f}s, valid_actions={t_valid_time:.2f}s, inv={t_inv_time:.2f}s)")
        
        return observation, reward, done, info
    
    def reset(self):
        """Reset the environment"""
        import time
        t0 = time.time()
        
        self.current_step = 0
        
        # Reset env
        t_reset_start = time.time()
        try:
            observation, info = self.env.reset()
        except Exception as e:
            print(f"[ERROR] JerichoWorker.reset: env.reset failed with error: {e}")
            raise
        t_reset_time = time.time() - t_reset_start
        
        # Add additional info
        info['game_name'] = self.game_name
        info['current_step'] = self.current_step
        info['max_score'] = self.env.get_max_score()
        
        # Get task description from game bindings
        bindings = self.env.bindings
        if bindings:
            info['task_description'] = f"Play the game '{bindings.get('name', self.game_name)}' and achieve the maximum score of {self.env.get_max_score()}."
        else:
            info['task_description'] = f"Play the interactive fiction game and achieve the maximum score of {self.env.get_max_score()}."
        
        # Get valid actions - THIS CAN BE SLOW!
        t_valid_start = time.time()
        try:
            info['valid'] = self.env.get_valid_actions()
        except Exception as e:
            info['valid'] = []
        t_valid_time = time.time() - t_valid_start
        
        # Get inventory
        t_inv_start = time.time()
        try:
            inventory = self.env.get_inventory()
            info['inv'] = ', '.join([obj.name for obj in inventory]) if inventory else 'empty'
        except Exception as e:
            info['inv'] = 'unknown'
        t_inv_time = time.time() - t_inv_start
        
        t_total = time.time() - t0
        print(f"[TIMING] JerichoWorker.reset: game='{self.game_name}', total={t_total:.2f}s "
              f"(reset={t_reset_time:.2f}s, valid_actions={t_valid_time:.2f}s, inv={t_inv_time:.2f}s)")
        
        return observation, info
    
    def get_valid_actions(self):
        """Get valid actions for the current state"""
        try:
            return self.env.get_valid_actions()
        except Exception:
            return []
    
    def get_score(self):
        """Get current score"""
        return self.env.get_score()
    
    def get_max_score(self):
        """Get maximum possible score"""
        return self.env.get_max_score()
    
    def look(self):
        """Look around"""
        obs, _, _, _ = self.env.step('look')
        return obs
    
    def inventory(self):
        """Check inventory"""
        obs, _, _, _ = self.env.step('inventory')
        return obs
    
    def close(self):
        """Close the environment"""
        if self.env is not None:
            self.env.close()


class JerichoEnvs(gym.Env):
    """
    Vectorized Jericho environment using Ray for parallelization.
    """
    
    def __init__(
        self,
        game_name: str,
        rom_path: str,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: Dict,
        is_train: bool = True,
        env_step_limit: int = 100
    ):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.game_name = game_name
        self.rom_path = rom_path
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        self.env_step_limit = env_step_limit
        
        # Create Ray remote actors
        print(f"[DEBUG] JerichoEnvs: Creating {self.num_processes} workers for game '{game_name}'...")
        env_worker = ray.remote(**resources_per_worker)(JerichoWorker)
        self.workers = []
        for i in range(self.num_processes):
            print(f"[DEBUG] JerichoEnvs: Spawning worker {i+1}/{self.num_processes}")
            worker = env_worker.remote(
                game_name=game_name,
                rom_path=rom_path,
                seed=seed + i,  # Each worker gets unique seed
                env_step_limit=env_step_limit,
                is_train=is_train
            )
            self.workers.append(worker)
        print(f"[DEBUG] JerichoEnvs: All workers spawned.")
        
        # Cache for valid actions per environment
        self.prev_valid_actions = [None for _ in range(self.num_processes)]
        self.prev_task_descriptions = [None for _ in range(self.num_processes)]
    
    def step(self, actions: List[str]):
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
        print(f"[DEBUG] JerichoEnvs.step: Starting with {len(actions)} actions")
        assert len(actions) == self.num_processes, \
            f"Number of actions ({len(actions)}) must equal number of processes ({self.num_processes})"
        
        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)
        
        # Collect results
        print(f"[DEBUG] JerichoEnvs.step: Waiting for {len(futures)} futures...")
        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []
        
        try:
            results = ray.get(futures, timeout=300)  # 5 minute timeout
        except Exception as e:
            print(f"[ERROR] JerichoEnvs.step: ray.get failed with error: {e}")
            raise
            
        print(f"[DEBUG] JerichoEnvs.step: Got {len(results)} results")
        for i, (obs, reward, done, info) in enumerate(results):
            text_obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            info_list.append(info)
            
            # Update cached valid actions
            self.prev_valid_actions[i] = info.get('valid', [])
        
        print(f"[DEBUG] JerichoEnvs.step: Completed, rewards sum={sum(rewards_list)}, dones count={sum(dones_list)}")
        return text_obs_list, rewards_list, dones_list, info_list
    
    def reset(self):
        """
        Reset all environments.
        
        Returns:
            text_obs_list: List of initial observation strings
            info_list: List of info dicts
        """
        print(f"[DEBUG] JerichoEnvs.reset: Starting reset for {self.num_processes} environments")
        text_obs_list = []
        info_list = []
        
        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)
        
        # Collect results
        print(f"[DEBUG] JerichoEnvs.reset: Waiting for {len(futures)} futures...")
        try:
            results = ray.get(futures, timeout=300)  # 5 minute timeout
        except Exception as e:
            print(f"[ERROR] JerichoEnvs.reset: ray.get failed with error: {e}")
            raise
            
        print(f"[DEBUG] JerichoEnvs.reset: Got {len(results)} results")
        for i, (obs, info) in enumerate(results):
            text_obs_list.append(obs)
            info_list.append(info)
            
            # Update cached valid actions and task descriptions
            self.prev_valid_actions[i] = info.get('valid', [])
            self.prev_task_descriptions[i] = info.get('task_description', '')
        
        print(f"[DEBUG] JerichoEnvs.reset: Completed successfully")
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


class JerichoMultiGameEnvs(gym.Env):
    """
    Vectorized Jericho environment that supports multiple games.
    Each reset can sample a different game from the provided list.
    """
    
    def __init__(
        self,
        game_names: List[str],
        rom_dir: str,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: Dict,
        is_train: bool = True,
        env_step_limit: int = 100
    ):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.game_names = game_names
        self.rom_dir = rom_dir
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        self.env_step_limit = env_step_limit
        self.seed = seed
        self.resources_per_worker = resources_per_worker
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Create Ray remote actors with random game assignments
        print(f"[DEBUG] JerichoMultiGameEnvs: Creating {self.num_processes} workers...")
        env_worker = ray.remote(**resources_per_worker)(JerichoWorker)
        self.workers = []
        self.current_games = []
        
        for i in range(self.num_processes):
            game_name = random.choice(game_names)
            self.current_games.append(game_name)
            rom_path = self._find_rom(game_name)
            
            print(f"[DEBUG] JerichoMultiGameEnvs: Spawning worker {i+1}/{self.num_processes} with game '{game_name}'")
            worker = env_worker.remote(
                game_name=game_name,
                rom_path=rom_path,
                seed=seed + i,
                env_step_limit=env_step_limit,
                is_train=is_train
            )
            self.workers.append(worker)
        print(f"[DEBUG] JerichoMultiGameEnvs: All workers spawned.")
        
        # Cache for valid actions per environment
        self.prev_valid_actions = [None for _ in range(self.num_processes)]
        self.prev_task_descriptions = [None for _ in range(self.num_processes)]
    
    def _find_rom(self, game_name: str) -> str:
        """Find ROM file for a game"""
        extensions = ['.z5', '.z8', '.z3', '.z4', '.z6', '']
        for ext in extensions:
            path = os.path.join(self.rom_dir, f"{game_name}{ext}")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"Could not find ROM for game '{game_name}' in {self.rom_dir}")
    
    def step(self, actions: List[str]):
        """Execute actions in all environments."""
        print(f"[DEBUG] JerichoMultiGameEnvs.step: Starting with {len(actions)} actions")
        assert len(actions) == self.num_processes
        
        futures = [worker.step.remote(action) for worker, action in zip(self.workers, actions)]
        print(f"[DEBUG] JerichoMultiGameEnvs.step: Waiting for {len(futures)} futures...")
        
        try:
            results = ray.get(futures, timeout=300)  # 5 minute timeout
        except Exception as e:
            print(f"[ERROR] JerichoMultiGameEnvs.step: ray.get failed with error: {e}")
            raise
        
        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []
        
        for i, (obs, reward, done, info) in enumerate(results):
            text_obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            info_list.append(info)
            self.prev_valid_actions[i] = info.get('valid', [])
        
        print(f"[DEBUG] JerichoMultiGameEnvs.step: Completed, rewards sum={sum(rewards_list)}, dones count={sum(dones_list)}")
        return text_obs_list, rewards_list, dones_list, info_list
    
    def reset(self):
        """Reset all environments. Reuse existing workers instead of recreating them."""
        import time
        t0 = time.time()
        
        # Just reset workers, don't recreate them (much faster!)
        t_reset_start = time.time()
        futures = [worker.reset.remote() for worker in self.workers]
        
        try:
            results = ray.get(futures, timeout=300)  # 5 minute timeout
        except Exception as e:
            print(f"[ERROR] JerichoMultiGameEnvs.reset: ray.get failed with error: {e}")
            raise
        t_reset_time = time.time() - t_reset_start
        
        text_obs_list = []
        info_list = []
        
        for i, (obs, info) in enumerate(results):
            text_obs_list.append(obs)
            info_list.append(info)
            self.prev_valid_actions[i] = info.get('valid', [])
            self.prev_task_descriptions[i] = info.get('task_description', '')
            # Update current games from info
            self.current_games[i] = info.get('game_name', self.current_games[i])
        
        t_total = time.time() - t0
        print(f"[TIMING] JerichoMultiGameEnvs.reset: TOTAL {t_total:.2f}s (reset={t_reset_time:.2f}s), "
              f"games={list(set(self.current_games))}")
        return text_obs_list, info_list
    
    @property
    def get_valid_actions(self):
        return self.prev_valid_actions
    
    @property
    def get_task_descriptions(self):
        return self.prev_task_descriptions
    
    def close(self):
        for worker in self.workers:
            ray.kill(worker)


def build_jericho_envs(
    game_name: str,
    rom_path: str,
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: Dict,
    is_train: bool = True,
    env_step_limit: int = 100
):
    """
    Build Jericho vectorized environments.
    
    Args:
        game_name: Name of the game (e.g., "zork1", "zork2"), or comma-separated
                   list for multi-game training (e.g., "zork1,zork2,zork3").
                   Special values:
                   - "all": Use all supported games (train split for training, test split for validation)
                   - "train": Use only training games
                   - "test": Use only test games
        rom_path: Path to the ROM file or directory containing ROM files
        seed: Random seed
        env_num: Number of environments
        group_n: Group size for rollout
        resources_per_worker: Ray resource configuration per worker
        is_train: Whether this is for training. When game_name="all":
                  - is_train=True: uses TRAIN_GAMES (44 games)
                  - is_train=False: uses TEST_GAMES (12 games)
        env_step_limit: Maximum steps per episode
        
    Returns:
        JerichoEnvs or JerichoMultiGameEnvs instance
        
    Note:
        The train/test split is designed for generalization evaluation:
        - Train on TRAIN_GAMES, test on TEST_GAMES to evaluate generalization to unseen games
        - Or use specific game names for in-distribution evaluation
    """
    # Check if multi-game mode
    if ',' in game_name or game_name.lower() in ['all', 'train', 'test']:
        if game_name.lower() == 'all':
            # Use train/test split based on is_train flag
            game_names = TRAIN_GAMES if is_train else TEST_GAMES
            print(f"[INFO] Using {'TRAIN' if is_train else 'TEST'} games split: {len(game_names)} games")
        elif game_name.lower() == 'train':
            game_names = TRAIN_GAMES
            print(f"[INFO] Using TRAIN games: {len(game_names)} games")
        elif game_name.lower() == 'test':
            game_names = TEST_GAMES
            print(f"[INFO] Using TEST games: {len(game_names)} games")
        else:
            game_names = [g.strip() for g in game_name.split(',')]
        
        return JerichoMultiGameEnvs(
            game_names=game_names,
            rom_dir=rom_path,
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            resources_per_worker=resources_per_worker,
            is_train=is_train,
            env_step_limit=env_step_limit
        )
    else:
        return JerichoEnvs(
            game_name=game_name,
            rom_path=rom_path,
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            resources_per_worker=resources_per_worker,
            is_train=is_train,
            env_step_limit=env_step_limit
        )



