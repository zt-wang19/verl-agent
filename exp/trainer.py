"""
自定义的 ALFWorld Trainer，继承自 RayPPOTrainer。
支持在 validate 时指定 repeat 次数，计算平均值和标准差。
"""

import json
import os
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

class SpidTrainer(RayPPOTrainer):
    """
    继承 RayPPOTrainer，添加 repeated validation 功能。
    """

    def _validate_repeated(self, num_iterations: int = 1):
        """
        多次运行 validate 并计算平均值和标准差。

        Args:
            num_iterations: validation 重复次数

        Returns:
            avg_metrics: 平均后的 metrics 字典
        """
        print("=" * 60)
        print(
            f"Running {num_iterations} validation iterations at step {self.global_steps}..."
        )
        print("=" * 60)

        # 保存原始输出目录
        original_dir = self.config.trainer.validation_data_dir

        # 存储每次迭代的 metrics
        all_metrics = []
        success_rate_keys = []  # 动态收集 success_rate 相关的 key

        for i in range(num_iterations):
            print("=" * 50)
            print(f"Validation iteration {i + 1}/{num_iterations}")
            print("=" * 50)

            # 设置当前迭代的输出目录
            # 非 val_only 模式下需要包含 global_steps 以区分不同训练步的 validation
            if original_dir is not None:
                if self.config.trainer.get("val_only", False):
                    # val_only 模式：直接用 iter_{i}
                    self.config.trainer.validation_data_dir = os.path.join(
                        original_dir, f"iter_{i}"
                    )
                else:
                    # 训练模式：用 step_{global_steps}/iter_{i} 区分不同训练步
                    self.config.trainer.validation_data_dir = os.path.join(
                        original_dir, f"step_{self.global_steps}", f"iter_{i}"
                    )
                os.makedirs(self.config.trainer.validation_data_dir, exist_ok=True)

            val_metrics = self._validate()
            all_metrics.append(val_metrics)

            # 收集 success_rate 相关的 key
            for key in val_metrics.keys():
                if "success_rate" in key.lower() and key not in success_rate_keys:
                    success_rate_keys.append(key)

            print(f"Iteration {i + 1} metrics:")
            pprint(val_metrics)

        # 恢复原始目录
        self.config.trainer.validation_data_dir = original_dir

        # 计算平均值
        print("=" * 60)
        print("SUMMARY: All iterations completed!")
        print("=" * 60)

        # 打印每次迭代的 success_rate
        print("--- Success Rate per Iteration ---")
        for i, metrics in enumerate(all_metrics):
            print(f"Iteration {i + 1}:")
            for key in success_rate_keys:
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")

        # 计算并打印平均值
        print("--- Average Success Rate ---")
        avg_metrics = {}
        for key in success_rate_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_value = np.mean(values)
                std_value = np.std(values)
                # 保留原始 key（用于 swanlab 曲线连续性）
                avg_metrics[key] = avg_value
                # 同时添加带后缀的 key
                avg_metrics[f"{key}/mean"] = avg_value
                avg_metrics[f"{key}/std"] = std_value
                print(f"  {key}: {avg_value:.4f} ± {std_value:.4f}")

        # 也计算其他 val/ 开头的指标的平均值
        print("--- Average of All Val Metrics ---")
        all_val_keys = set()
        for metrics in all_metrics:
            for key in metrics.keys():
                if key.startswith("val/"):
                    all_val_keys.add(key)

        for key in sorted(all_val_keys):
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_value = np.mean(values)
                std_value = np.std(values)
                # 保留原始 key（用于 swanlab 曲线连续性）
                avg_metrics[key] = avg_value
                # 同时添加带后缀的 key
                avg_metrics[f"{key}/mean"] = avg_value
                avg_metrics[f"{key}/std"] = std_value
                print(f"  {key}: {avg_value:.4f} ± {std_value:.4f}")

        print("=" * 60)
        print("Repeated validation completed!")
        if original_dir:
            print(f"Trajectories saved to: {original_dir}")
        print("=" * 60)

        return avg_metrics

    def _validate(self):
        """
        重写 _validate 方法，添加验证后轨迹数据的保存功能。
        """
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        # 用于保存轨迹数据的列表
        all_trajectories = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Note: input_texts will be collected after multi_turn_loop since batch size may change

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.val_envs,
                is_train=False,
            )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            
            # 从 output batch 获取输入文本（multi_turn_loop 后 batch 大小可能改变）
            if "prompts" in test_output_gen_batch.batch:
                current_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_output_gen_batch.batch["prompts"]]
            else:
                # 如果没有 prompts 字段，用 unknown 填充
                current_input_texts = ["unknown"] * len(output_texts)
            sample_inputs.extend(current_input_texts)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            
            # 收集轨迹数据用于保存
            batch_size = len(output_texts)
            for i in range(batch_size):
                traj_data = {
                    "input": current_input_texts[i] if i < len(current_input_texts) else "unknown",
                    "output": output_texts[i],
                    "score": scores[i],
                    "traj_uid": str(test_output_gen_batch.non_tensor_batch['traj_uid'][i]),
                    "tool_callings": int(test_output_gen_batch.non_tensor_batch['tool_callings'][i]),
                }
                # 添加 data_source
                data_sources = test_batch.non_tensor_batch.get('data_source', ['unknown'] * batch_size)
                traj_data["data_source"] = str(data_sources[i]) if i < len(data_sources) else "unknown"
                
                # 添加其他可用的 non_tensor_batch 字段
                for key in test_batch.non_tensor_batch.keys():
                    if key not in ['data_source', 'traj_uid', 'tool_callings'] and 'success_rate' not in key:
                        try:
                            val = test_batch.non_tensor_batch[key][i]
                            # 尝试转换为可序列化的类型
                            if isinstance(val, np.ndarray):
                                val = val.tolist()
                            elif isinstance(val, (np.integer, np.floating)):
                                val = val.item()
                            elif isinstance(val, np.bool_):
                                val = bool(val)
                            traj_data[key] = val
                        except (IndexError, TypeError):
                            pass
                
                all_trajectories.append(traj_data)
            
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # 保存轨迹数据
        validation_data_dir = self.config.trainer.get("validation_data_dir", None)
        if validation_data_dir is not None:
            os.makedirs(validation_data_dir, exist_ok=True)
            traj_file = os.path.join(validation_data_dir, f"trajectories_step_{self.global_steps}.jsonl")
            with open(traj_file, "w", encoding="utf-8") as f:
                for traj in all_trajectories:
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            print(f"Saved {len(all_trajectories)} trajectories to {traj_file}")
            
            # 同时保存一个汇总信息
            summary_file = os.path.join(validation_data_dir, f"summary_step_{self.global_steps}.json")
            summary = {
                "global_step": self.global_steps,
                "num_trajectories": len(all_trajectories),
                "success_rate": success_rate,
                "mean_score": float(reward_tensor.mean().item()),
                "std_score": float(reward_tensor.std().item()),
            }
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"Saved summary to {summary_file}")

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        return metric_dict

    def fit(self):
        """
        重写 fit 方法，支持 repeated validation。
        """
        from copy import deepcopy

        import ray
        import torch
        from omegaconf import OmegaConf
        from tqdm import tqdm

        from agent_system.multi_turn_rollout import adjust_batch
        from gigpo import core_gigpo
        from verl import DataProto
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.metric_utils import (
            compute_data_metrics,
            compute_throughout_metrics,
            compute_timing_metrics,
        )
        from verl.trainer.ppo.ray_trainer import (
            AdvantageEstimator,
            _timer,
            apply_invalid_action_penalty,
            apply_kl_penalty,
            compute_advantage,
            compute_response_mask,
        )
        from verl.trainer.ppo.reward import compute_reward, compute_reward_async
        from verl.utils.metric import reduce_metrics
        from verl.utils.seqlen_balancing import (
            get_seqlen_balanced_partitions,
            log_seqlen_unbalance,
        )
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # 获取 validation repeat 次数
        num_val_iterations = self.config.trainer.get("num_val_iterations", 1)

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            if num_val_iterations > 1:
                val_metrics = self._validate_repeated(num_val_iterations)
            else:
                val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            print("Initial validation metrics:")
            pprint(val_metrics)
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        ################ agent-environment loop ###############
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            )

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = (
                            core_gigpo.compute_step_discounted_returns(
                                batch=batch, gamma=self.config.algorithm.gamma
                            )
                        )
                        batch.batch["step_rewards"] = step_rewards_tensor

                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                batch, self.config, self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn
                            )

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_loss = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy_loss": entropy_loss.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(
                                rollout_probs_diff, response_mask.bool()
                            )
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                            else:
                                ref_log_prob = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch)
                                )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(
                                future_reward
                            )
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {
                                    k: np.array(v)
                                    for k, v in reward_extra_infos_dict.items()
                                }
                            )

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get(
                            "use_invalid_action_penalty", True
                        ):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        # compute advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(f"batch.batch.keys(): {batch.batch.keys()}")
                            inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores = (
                                batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate - 使用 repeated validation
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            if num_val_iterations > 1:
                                val_metrics: dict = self._validate_repeated(
                                    num_val_iterations
                                )
                            else:
                                val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    print("Final validation metrics:")
                    pprint(last_val_metrics)
                    progress_bar.close()
                    return
