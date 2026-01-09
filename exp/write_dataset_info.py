"""
数据集划分脚本：
- 类型：inverse_dynamics_only, state_prediction_only, mixed (0.5:0.5)
- 数据量：100 steps, 1000 steps, 10000 steps (batch_size=32)
- 共 3*3=9 个数据集
"""

import json
import os
import random
from pathlib import Path

# ============ 可修改的环境名称 ============
ENV_NAME = "webshop"  # 可选: "alfworld", "webshop", 等

# 配置
BATCH_SIZE = 32
STEPS_LIST = [50, 100, 200, 500]  # 50, 100, 200, 500, 1k, 10k steps
DATA_TYPES = ["inverse_dynamics_only", "state_prediction_only", "mixed"]

# 路径配置
DATA_DIR = Path(f"data/exp/{ENV_NAME}_spid")
OUTPUT_DIR = DATA_DIR / "llamafactory"
DATASET_INFO_PATH = Path(
    "data/exp/dataset_info.json"
)

# 源文件
INVERSE_DYNAMICS_FILE = DATA_DIR / "inverse_dynamics_hl2.jsonl"
STATE_PREDICTION_FILE = DATA_DIR / "state_prediction_hl2.jsonl"

# 随机种子
RANDOM_SEED = 42


def load_jsonl(file_path: Path) -> list:
    """加载 jsonl 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: list, file_path: Path):
    """保存为 jsonl 文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {file_path}")


def create_dataset_entry(file_name: str) -> dict:
    """创建 dataset_info.json 中的条目"""
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def main():
    random.seed(RANDOM_SEED)

    print("Loading source data...")
    inverse_dynamics_data = load_jsonl(INVERSE_DYNAMICS_FILE)
    state_prediction_data = load_jsonl(STATE_PREDICTION_FILE)

    print(f"Inverse dynamics samples: {len(inverse_dynamics_data)}")
    print(f"State prediction samples: {len(state_prediction_data)}")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载或初始化 dataset_info.json
    if DATASET_INFO_PATH.exists():
        with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    # 生成数据集
    for steps in STEPS_LIST:
        num_samples = steps * BATCH_SIZE
        print(f"\n{'='*60}")
        print(f"Generating datasets for {steps} steps ({num_samples} samples)")
        print(f"{'='*60}")

        for data_type in DATA_TYPES:
            # 生成文件名和数据集名
            dataset_name = f"{ENV_NAME}_spid_{data_type}_{steps}steps"
            file_name = f"{data_type}_{steps}steps.jsonl"
            file_path = OUTPUT_DIR / file_name

            if data_type == "inverse_dynamics_only":
                # 只使用 inverse dynamics 数据
                sampled_data = random.sample(inverse_dynamics_data, num_samples)

            elif data_type == "state_prediction_only":
                # 只使用 state prediction 数据
                sampled_data = random.sample(state_prediction_data, num_samples)

            elif data_type == "mixed":
                # 0.5:0.5 混合
                half_samples = num_samples // 2
                inverse_samples = random.sample(inverse_dynamics_data, half_samples)
                state_samples = random.sample(
                    state_prediction_data, num_samples - half_samples
                )
                sampled_data = inverse_samples + state_samples
                random.shuffle(sampled_data)  # 打乱顺序

            # 保存数据
            save_jsonl(sampled_data, file_path)

            # 更新 dataset_info
            relative_path = f"{ENV_NAME}_spid/llamafactory/{file_name}"
            dataset_info[dataset_name] = create_dataset_entry(relative_path)
            print(f"  -> {dataset_name}: {len(sampled_data)} samples")

    # 保存更新后的 dataset_info.json
    DATASET_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated dataset_info.json at {DATASET_INFO_PATH}")

    # 打印汇总
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total datasets created: {len(STEPS_LIST) * len(DATA_TYPES)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Dataset info: {DATASET_INFO_PATH}")
    print("\nDataset names:")
    for steps in STEPS_LIST:
        for data_type in DATA_TYPES:
            dataset_name = f"{ENV_NAME}_spid_{data_type}_{steps}steps"
            print(f"  - {dataset_name}")


if __name__ == "__main__":
    main()
