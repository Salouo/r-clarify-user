import json
import copy
from pathlib import Path


def expand_reflective_actions(data):
    """
    Expand reflective_actions (a list) in each sample into multiple samples,
    each keeping a single reflective_action (string).

    Assumed top-level structure:
    {
      "n_sampels": ...,
      "samples": [ {...}, {...}, ... ]
    }
    """

    if not isinstance(data, dict):
        raise ValueError("顶层 JSON 结构应该是一个 dict（包含 'samples' 字段）。")

    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("data['samples'] 必须是一个列表。")

    new_samples = []
    new_index = 0

    for sample in samples:
        # Deep copy to avoid cross-sample mutation.
        reflective_actions = sample.get("reflective_actions")

        # Skip if reflective_actions is missing or empty (adjust if you want to keep originals).
        if not reflective_actions:
            continue

        for action in reflective_actions:
            # Copy the original sample.
            new_sample = copy.deepcopy(sample)

            # Update index.
            new_sample["index"] = new_index
            new_index += 1

            # Replace reflective_actions -> reflective_action.
            new_sample["reflective_action"] = action
            if "reflective_actions" in new_sample:
                del new_sample["reflective_actions"]

            new_samples.append(new_sample)

    # Update top-level fields.
    data["samples"] = new_samples
    data["n_sampels"] = len(new_samples)

    return data


def main():
    # Adjust paths as needed.
    input_path_str = "data/processed_data2.json"
    output_path_str = "data/processed_data_expanded.json"

    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    # Read JSON.
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Transform data.
    transformed = expand_reflective_actions(data)

    # Write JSON.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"处理完成，结果已保存到: {output_path}")
    print(f"新的样本数 n_sampels = {transformed.get('n_sampels')}")


if __name__ == "__main__":
    main()
