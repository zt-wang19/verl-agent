import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def format_history_pairs(
    history_pairs: List[Tuple[str, str]],
    history_start_step: int,
    total_previous_steps: int,
) -> str:
    if total_previous_steps == 0:
        return "You have not taken any previous steps before this action.\n"
    lines = [
        f'[Observation {history_start_step + idx}: "{obs}", Action {history_start_step + idx}: "{act}"]'
        for idx, (obs, act) in enumerate(history_pairs)
    ]
    return (
        f"Prior to this step, you have already taken {total_previous_steps} step(s). "
        "Below are the history observations and the corresponding actions you took:\n"
        + "\n".join(lines)
        + "\n"
    )


def format_available_actions(actions: List[str]) -> str:
    if not actions:
        return (
            "The environment did not provide any admissible actions for this step; "
            "reason about plausible actions based on the observations.\n"
        )
    choices = "\n".join(f"- {act}" for act in actions)
    return (
        "You must select exactly one of the following admissible actions to explain the transition:\n"
        f"{choices}\n"
    )


def create_state_prediction_messages(
    history_pairs: List[Tuple[str, str]],
    current_obs: str,
    action: str,
    next_obs: str,
    step_number: int,
    task: str,
    history_start_step: int,
) -> List[Dict[str, str]]:
    total_previous_steps = step_number - 1
    history_text = format_history_pairs(
        history_pairs, history_start_step, total_previous_steps
    )
    user_content = (
        "You are an expert agent operating in the WebShop Environment.\n"
        f"Your task is: {task}\n"
        f"{history_text}"
        f"You are now at step {step_number} and your current observation is: {current_obs}\n"
        f"You take the action: {action}.\n\n"
        "Please predict the observation after taking this action.\n"
        "Present your prediction within <observation> </observation> tags."
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"<observation>{next_obs}</observation>"},
    ]


def create_inverse_dynamics_messages(
    history_pairs: List[Tuple[str, str]],
    current_obs: str,
    next_obs: str,
    action: str,
    admissible_actions: List[str],
    step_number: int,
    task: str,
    history_start_step: int,
) -> List[Dict[str, str]]:
    total_previous_steps = step_number - 1
    history_text = format_history_pairs(
        history_pairs, history_start_step, total_previous_steps
    )
    available_actions_text = format_available_actions(admissible_actions)
    user_content = (
        "You are an expert agent operating in the WebShop Environment.\n"
        f"Your task is: {task}\n"
        f"{history_text}"
        f"You are now at step {step_number}.\n"
        f"Your current observation is: {current_obs}\n"
        f"In the next timestep, the environment observation becomes: {next_obs}\n\n"
        f"{available_actions_text}"
        "Choose the single admissible action that best explains this transition.\n"
        "Present your chosen action within <action> </action> tags."
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"<action>{action}</action>"},
    ]


def process_trajectory(
    traj: Dict[str, Any], history_length: int | None = None
) -> Tuple[List[Dict], List[Dict]]:
    task = traj.get("task", "")
    obs_list = traj["obs"]
    actions_list = traj["actions"]
    admissibles_list = traj["admissibles"]

    state_samples = []
    inverse_samples = []
    history_pairs: List[Tuple[str, str]] = []

    # 遍历每个(obs, action)对，预测下一个obs
    for i in range(len(actions_list)):
        current_obs = obs_list[i].strip()
        action = actions_list[i].strip()
        next_obs = obs_list[i + 1].strip() if i + 1 < len(obs_list) else ""
        admissible = admissibles_list[i] if i < len(admissibles_list) else []

        if not current_obs or not action or not next_obs:
            continue

        step_number = len(history_pairs) + 1

        # 根据 history_length 截取历史记录
        if history_length is not None:
            truncated_history = list(history_pairs[-history_length:])
        else:
            truncated_history = list(history_pairs)

        # 计算历史记录的起始步数
        history_start_step = step_number - len(truncated_history)

        # State prediction
        state_msgs = create_state_prediction_messages(
            truncated_history,
            current_obs,
            action,
            next_obs,
            step_number,
            task,
            history_start_step,
        )
        state_samples.append({"messages": state_msgs})

        # Inverse dynamics
        inverse_msgs = create_inverse_dynamics_messages(
            truncated_history,
            current_obs,
            next_obs,
            action,
            admissible,
            step_number,
            task,
            history_start_step,
        )
        inverse_samples.append({"messages": inverse_msgs})

        history_pairs.append((current_obs, action))

    return state_samples, inverse_samples


def process_data(
    input_path: str, output_dir: str, history_length: int | None = None
) -> None:
    """
    读取轨迹数据，生成 state prediction 和 inverse dynamics 样本并保存。

    Args:
        input_path: 输入 JSON 文件路径
        output_dir: 输出目录路径
        history_length: 历史记录的最大长度，None 表示不限制
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_state_samples = []
    all_inverse_samples = []

    for traj in data:
        state_samples, inverse_samples = process_trajectory(traj, history_length)
        all_state_samples.extend(state_samples)
        all_inverse_samples.extend(inverse_samples)

    os.makedirs(output_dir, exist_ok=True)

    hl_suffix = f"_hl{history_length}" if history_length is not None else ""
    state_path = os.path.join(output_dir, f"state_prediction{hl_suffix}.jsonl")
    inverse_path = os.path.join(output_dir, f"inverse_dynamics{hl_suffix}.jsonl")

    with open(state_path, "w", encoding="utf-8") as f:
        for sample in all_state_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(inverse_path, "w", encoding="utf-8") as f:
        for sample in all_inverse_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"State prediction samples: {len(all_state_samples)} -> {state_path}")
    print(f"Inverse dynamics samples: {len(all_inverse_samples)} -> {inverse_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--history_length",
        "-hl",
        type=int,
        default=2,
        help="Maximum history length (default: None, no limit)",
    )
    args = parser.parse_args()
    process_data(args.input, args.output_dir, args.history_length)


if __name__ == "__main__":
    main()

