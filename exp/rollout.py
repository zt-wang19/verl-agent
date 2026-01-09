
from agent_system.multi_turn_rollout import TrajectoryCollector
from verl import DataProto
import torch
import numpy as np

class SpidTrajectoryCollector(TrajectoryCollector):
    def preprocess_single_sample(self, item, gen_batch, obs):
        row_dict = super().preprocess_single_sample(item, gen_batch, obs)
        
        # Add 'admissibles' and 'history' to row_dict if present in obs
        if 'admissibles' in obs:
            row_dict['admissibles'] = obs['admissibles'][item]
        if 'history' in obs:
            row_dict['history'] = obs['history'][item]
            
        return row_dict

    def gather_rollout_data(
            self,
            total_batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
            ):
        
        # Inject next_obs into total_batch_list
        for env_traj in total_batch_list:
            for i in range(len(env_traj) - 1):
                # env_traj[i] is step t. env_traj[i+1] is step t+1.
                # anchor_obs in env_traj[i+1] is the next_obs for step t.
                env_traj[i]['next_obs'] = env_traj[i+1]['anchor_obs']
            
            # Last step next_obs
            if env_traj:
                env_traj[-1]['next_obs'] = "" # Use empty string or None. Process_state_prediction handles it.

        return super().gather_rollout_data(
            total_batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings
        )
