
import torch
import numpy as np
import itertools
import os
import json
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.utils.torch_functional import logprobs_from_logits
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.device import get_torch_device

from exp.aux_loss_utils import create_state_prediction_messages, create_inverse_dynamics_messages

import logging
logger = logging.getLogger(__file__)

class SpidPPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        self.sp_coef = config.get("sp_coef", 0.0) 
        self.id_coef = config.get("id_coef", 0.0)
        self.aux_history_length = config.get("aux_history_length", None)
        self.aux_max_length = config.get("aux_max_length", None)
        self.tokenizer = None
        self.debug_dir = None # Set by worker or config if needed
        self.global_step = 0 # Track step for debug filename

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_debug_dir(self, debug_dir):
        self.debug_dir = debug_dir
        if self.debug_dir and not os.path.exists(self.debug_dir):
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
            except Exception:
                pass

    def compute_auxiliary_loss(self, data: dict, temperature=1.0):
        if self.tokenizer is None:
            return 0.0, {}

        # If both coefficients are 0 and no debug is needed, skip everything
        if self.sp_coef <= 0 and self.id_coef <= 0 and not self.debug_dir:
            return 0.0, {}

        history_batch = data.get("history")
        if history_batch is None:
            return 0.0, {}
        
        anchor_obs_batch = data.get("anchor_obs")
        next_obs_batch = data.get("next_obs")
        admissibles_batch = data.get("admissibles")
        
        responses = data.get("responses")
        action_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        sp_inputs = []
        id_inputs = []
        
        # Debug data collection
        debug_data = []

        device = get_torch_device().current_device()
        batch_size = len(history_batch)
        
        for i in range(batch_size):
            history_list = history_batch[i] 
            current_obs = str(anchor_obs_batch[i]) if anchor_obs_batch[i] is not None else ""
            action_text = action_texts[i].strip()
            next_obs = str(next_obs_batch[i]) if next_obs_batch[i] is not None else ""
            admissibles = admissibles_batch[i] if admissibles_batch is not None else []
            
            # Parse and validate action from action_text
            action_text_lower = action_text.lower()
            start_tag = "<action>"
            end_tag = "</action>"
            start_idx = action_text_lower.find(start_tag)
            end_idx = action_text_lower.find(end_tag)
            
            # Check if action is valid
            is_valid = True
            if start_idx == -1 or end_idx == -1:
                is_valid = False
            
            
            # Extract the action from the tags
            action = action_text_lower[start_idx + len(start_tag):end_idx].strip()
            
            if not next_obs:
                continue

            history_pairs = [(h['text_obs'], h['action']) for h in history_list]
            total_history_steps = len(history_pairs)
            step_number = total_history_steps + 1
            
            # Truncate history if needed
            if self.aux_history_length is not None and len(history_pairs) > self.aux_history_length:
                history_pairs = history_pairs[-self.aux_history_length:]
                
            history_start_step = total_history_steps - len(history_pairs) + 1
            
            # Use generic prompt or try to detect environment type? 
            # For now using default system prompt in utils.
            
            # --- State Prediction ---
            sp_text = ""
            if self.sp_coef > 0 or self.debug_dir:
                sp_msgs = create_state_prediction_messages(
                    history_pairs=history_pairs, 
                    current_obs=current_obs, 
                    action=action, 
                    next_obs=next_obs, 
                    step_number=step_number, 
                    history_start_step=history_start_step,
                    task=None
                )
                sp_text = self.tokenizer.apply_chat_template(sp_msgs, tokenize=False, add_generation_prompt=False)
                if self.sp_coef > 0:
                    sp_inputs.append(sp_text)

            # --- Inverse Dynamics ---
            id_text = ""
            if self.id_coef > 0 or self.debug_dir:
                id_msgs = create_inverse_dynamics_messages(
                    history_pairs=history_pairs, 
                    current_obs=current_obs, 
                    next_obs=next_obs, 
                    action=action, 
                    admissible_actions=admissibles, 
                    step_number=step_number, 
                    history_start_step=history_start_step,
                    task=None
                )
                id_text = self.tokenizer.apply_chat_template(id_msgs, tokenize=False, add_generation_prompt=False)
                if self.id_coef > 0:
                    id_inputs.append(id_text)
            
            if self.debug_dir:
                raw_data_i = {}
                for k, v in data.items():
                    try:
                        # Attempt to extract the i-th element if it looks like a batch
                        if (isinstance(v, (list, tuple, np.ndarray)) or isinstance(v, torch.Tensor)) and len(v) == batch_size:
                            val = v[i]
                        else:
                            val = v
                        
                        # Convert to serializable format
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().tolist()
                        elif isinstance(val, np.ndarray):
                            val = val.tolist()
                        
                        # Handle list containing tensors (e.g. history might not have tensors but just in case)
                        # We assume history is json-serializable (list of dicts of strings)
                        
                        raw_data_i[k] = val
                    except Exception as e:
                        raw_data_i[k] = f"<Error extracting {k}: {e}>"

                debug_data.append({
                    "step_number": step_number,
                    "history_len": len(history_pairs),
                    "current_obs": current_obs,
                    "action": action,
                    "next_obs": next_obs,
                    "admissibles_count": len(admissibles),
                    "sp_input": sp_text,
                    "id_input": id_text,
                    "raw_data": raw_data_i
                })

        # Save debug data if enabled and available
        if self.debug_dir and debug_data:
            rank = int(os.environ.get("RANK", 0))
            debug_file = os.path.join(self.debug_dir, f"aux_debug_rank{rank}.jsonl")
            try:
                with open(debug_file, "a", encoding="utf-8") as f:
                    for item in debug_data:
                        item["global_step"] = self.global_step
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write debug data: {e}")

        if not sp_inputs and not id_inputs:
            return 0.0, {}

        def compute_lm_loss(texts):
            max_len = self.aux_max_length if self.aux_max_length is not None else 2560

            # Chunking to avoid OOM
            aux_mb = self.config.get("aux_micro_batch_size", 8)
            total_loss = 0.0
            total_samples = 0
            
            for i in range(0, len(texts), aux_mb):
                chunk = texts[i : i + aux_mb]
                if not chunk: continue
                
                encodings = self.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
                input_ids = encodings['input_ids'].to(device)
                
                if input_ids.size(1) == 0:
                    continue

                attention_mask = encodings['attention_mask'].to(device)
                labels = input_ids.clone()
                
                outputs = self.actor_module(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                # Accumulate loss weighted by batch size
                total_loss += outputs.loss * len(chunk)
                total_samples += len(chunk)

            if total_samples == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            return total_loss / total_samples

        metrics = {}
        total_aux_loss = 0.0

        if self.sp_coef > 0 and sp_inputs:
            sp_loss = compute_lm_loss(sp_inputs)
            total_aux_loss += self.sp_coef * sp_loss
            metrics['actor/sp_loss'] = sp_loss.detach().item()

        if self.id_coef > 0 and id_inputs:
            id_loss = compute_lm_loss(id_inputs)
            total_aux_loss += self.id_coef * id_loss
            metrics['actor/id_loss'] = id_loss.detach().item()

        return total_aux_loss, metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        
        # Track global step if provided in data? 
        # Usually metrics['training/global_step'] is available but we are in actor.
        # We can increment a local counter or try to find it.
        self.global_step += 1

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        non_tensor_select_keys = ["history", "anchor_obs", "next_obs", "admissibles", "raw_prompt"]

        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
            
        data_selected = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys.append("multi_modal_inputs")
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            num_mini_batches = max(1, data_selected.batch.batch_size[0] // self.config.ppo_mini_batch_size)
            dataloader = data_selected.chunk(num_mini_batches)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data_chunk in enumerate(dataloader):
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = data_chunk.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data_chunk.chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                     max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                     micro_batches, _ = rearrange_micro_batches(batch=data_chunk.batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = max(1, data_chunk.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu)
                    micro_batches = data_chunk.chunk(num_micro_batches)

                self.actor_optimizer.zero_grad()

                for data_mb in micro_batches:
                    if isinstance(data_mb, DataProto):
                        data_dict = {**data_mb.batch.to(get_torch_device().current_device()), **data_mb.non_tensor_batch}
                    else:
                        data_dict = data_mb.to(get_torch_device().current_device())

                    responses = data_dict["responses"]
                    response_length = responses.size(1)
                    attention_mask = data_dict["attention_mask"]
                    if multi_turn:
                        response_mask = data_dict["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data_dict["old_log_probs"]
                    advantages = data_dict["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data_dict, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data_dict["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()

                    aux_loss, aux_metrics = self.compute_auxiliary_loss(data_dict, temperature)
                    if aux_loss != 0:
                        policy_loss = policy_loss + aux_loss
                        metrics.update(aux_metrics)

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(data_dict) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data_metric = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data_metric)

                grad_norm = self._optimizer_step()
                data_metric = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data_metric)
        self.actor_optimizer.zero_grad()
        return metrics

class SpidActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils.import_utils import import_external_libs
        from omegaconf import OmegaConf, open_dict
        from verl.workers.actor import DataParallelPPOActor
        from verl.utils.fs import copy_to_local
        from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu, offload_fsdp_optimizer, fsdp_version
        
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get('use_shm', False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            
            self.actor = SpidPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)
            
            if hasattr(self, 'tokenizer'):
                self.actor.set_tokenizer(self.tokenizer)
            
            # Use debug_dir if configured in actor config or use a default relative to working dir
            # Note: config.actor is DictConfig.
            debug_dir = self.config.actor.get("debug_dir", "debug_data")
            self.actor.set_debug_dir(debug_dir)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            from verl.utils.flops_counter import FlopsCounter
            from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
            
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )
