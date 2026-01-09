
import os
from functools import partial
from omegaconf import OmegaConf

class SpidMixin:
    """
    Mixin to inject admissibles and history into observation.
    Expects self.envs to provide get_admissible_commands or equivalent,
    and self.memory to store history.
    """
    def _augment_obs(self, obs, infos=None):
        # We need to extract admissibles.
        # Different environments have different ways to get valid actions.
        # ScienceWorld/TextWorldExpress pass 'valid' in infos.
        # AlfWorld provides get_admissible_commands on envs.
        
        admissibles = None
        if hasattr(self, 'envs') and hasattr(self.envs, 'get_admissible_commands'):
            admissibles = self.envs.get_admissible_commands
        elif infos is not None:
             # ScienceWorld/TextWorldExpress puts valid actions in infos
             # infos is a list of dicts
             admissibles = [info.get('valid', []) for info in infos]
             
             # Webshop puts available_actions in infos
             if not admissibles or not any(admissibles):
                 admissibles = []
                 for info in infos:
                     if 'available_actions' in info:
                         # Webshop format
                         # We need to format it if not already formatted?
                         # WebshopEnvironmentManager.format_avail_actions does it but returns list
                         # We can just pass the raw dict or try to format?
                         # For consistency with rollout.py which expects list of strings (usually),
                         # let's try to get a list.
                         # But for now let's focus on list of strings.
                         # If it's Webshop, available_actions is a dict.
                         pass

        if admissibles is None:
            # Default to empty list
            num_envs = len(obs['text']) if obs.get('text') else 0
            admissibles = [[] for _ in range(num_envs)]
            
        obs['admissibles'] = admissibles
        
        # History
        # self.memory[i] is the history list
        num_envs = len(admissibles)
        if hasattr(self, 'memory'):
             obs['history'] = [list(self.memory[i]) for i in range(num_envs)]
        
        return obs

def _wrap_manager(BaseManager):
    class SpidManager(BaseManager, SpidMixin):
        def reset(self, kwargs):
            obs, infos = super().reset(kwargs)
            return self._augment_obs(obs, infos), infos

        def step(self, text_actions):
            next_obs, rewards, dones, infos = super().step(text_actions)
            return self._augment_obs(next_obs, infos), rewards, dones, infos
    return SpidManager

def make_envs(config):
    """
    Custom make_envs to use Spid EnvironmentManagers
    """
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    env_name_lower = config.env.env_name.lower()

    if "alfworld" in env_name_lower:
        from agent_system.environments.env_manager import AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")
        
        if not os.path.exists(alf_config_path):
             alf_config_path = 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, 
        }
        
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        SpidAlfWorldEnvironmentManager = _wrap_manager(AlfWorldEnvironmentManager)
        envs = SpidAlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = SpidAlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    
    elif "scienceworld" in env_name_lower:
        from agent_system.environments.env_manager import ScienceWorldEnvironmentManager
        from agent_system.environments.env_package.scienceworld import build_scienceworld_envs, scienceworld_projection
        
        task_name = config.env.scienceworld.task_name
        simplification_str = getattr(config.env.scienceworld, 'simplification_str', 'easy')
        env_step_limit = getattr(config.env.scienceworld, 'env_step_limit', 100)
        
        _envs = build_scienceworld_envs(
            task_name=task_name,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            resources_per_worker=resources_per_worker,
            is_train=True,
            simplification_str=simplification_str,
            env_step_limit=env_step_limit
        )
        _val_envs = build_scienceworld_envs(
            task_name=task_name,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            resources_per_worker=resources_per_worker,
            is_train=False,
            simplification_str=simplification_str,
            env_step_limit=env_step_limit
        )
        
        projection_f = partial(scienceworld_projection)
        SpidScienceWorldEnvironmentManager = _wrap_manager(ScienceWorldEnvironmentManager)
        envs = SpidScienceWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = SpidScienceWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
        
    elif "textworld_express" in env_name_lower:
        from agent_system.environments.env_manager import TextWorldExpressEnvironmentManager
        from agent_system.environments.env_package.textworld_express import build_textworld_express_envs, textworld_express_projection
        
        game_name = getattr(config.env.textworld_express, 'game_name', 'cookingworld')
        env_step_limit = getattr(config.env.textworld_express, 'env_step_limit', 50)
        game_params = getattr(config.env.textworld_express, 'game_params', '')
        
        _envs = build_textworld_express_envs(
            game_name=game_name,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            resources_per_worker=resources_per_worker,
            is_train=True,
            env_step_limit=env_step_limit,
            game_params=game_params
        )
        _val_envs = build_textworld_express_envs(
            game_name=game_name,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            resources_per_worker=resources_per_worker,
            is_train=False,
            env_step_limit=env_step_limit,
            game_params=game_params
        )
        
        projection_f = partial(textworld_express_projection)
        SpidTextWorldExpressEnvironmentManager = _wrap_manager(TextWorldExpressEnvironmentManager)
        envs = SpidTextWorldExpressEnvironmentManager(_envs, projection_f, config)
        val_envs = SpidTextWorldExpressEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs

    else:
        # Fallback to original make_envs
        from agent_system.environments import make_envs as original_make_envs
        return original_make_envs(config)
