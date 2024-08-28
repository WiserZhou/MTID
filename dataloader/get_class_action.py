import os
import numpy as np
import json

def process_files_in_directory_NIV(directory, prefixes, output_file):
    all_unique_steps = {}  # 用于存储每个前缀的唯一 steps_ids

    # 遍历每个前缀
    for idx, prefix in enumerate(prefixes):
        unique_steps = set()  # 用于存储唯一的 steps_ids

        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            if filename.startswith(prefix) and filename.endswith('.npy'):
                filepath = os.path.join(directory, filename)
                try:
                    # 从 .npy 文件加载数据
                    data = np.load(filepath, allow_pickle=True)

                    # 如果数据是字典，提取 'steps_ids'
                    if isinstance(data, dict):
                        steps_ids = data.get('steps_ids', [])
                    else:
                        steps_ids = data
                    
                    # 将 steps_ids 的元素添加到集合中以去除重复项
                    unique_steps.update(np.array(steps_ids).flatten())
                except Exception as e:
                    print(f"无法处理文件 {filepath}: {e}")

        # 将 numpy 类型转换为 Python 本地类型以进行 JSON 序列化
        all_unique_steps[idx] = [int(step) for step in unique_steps]

    # 将字典保存为 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(all_unique_steps, f, indent=4)
    
    print(f"所有前缀的唯一步骤已保存到 {output_file}")
    
def process_files_in_directory_coin(directory, output_file):
    all_unique_steps = {}  # 用于存储每个 cls 的唯一 steps_ids

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            filepath = os.path.join(directory, filename)
            try:
                # 从 .npy 文件加载数据
                data = np.load(filepath, allow_pickle=True)
                
                # 提取 cls 和 steps_ids
                cls = data.get('cls', None)
                steps_ids = data.get('steps_ids', [])
                
                if cls is not None:
                    # 确保 steps_ids 是一个数组并将其展平
                    steps_ids = np.array(steps_ids).flatten()
                    
                    if cls not in all_unique_steps:
                        all_unique_steps[cls] = set()
                    
                    # 将 steps_ids 添加到该 cls 的集合中
                    all_unique_steps[cls].update(steps_ids)
            
            except Exception as e:
                print(f"无法处理文件 {filepath}: {e}")

    # 将 numpy 类型转换为 Python 本地类型以进行 JSON 序列化
    all_unique_steps = {cls: list(map(int, ids)) for cls, ids in all_unique_steps.items()}

    # 根据 cls 键对字典进行排序
    sorted_all_unique_steps = dict(sorted(all_unique_steps.items()))

    # 将排序后的字典保存为 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(sorted_all_unique_steps, f, indent=4)
    
    print(f"所有 cls 值的唯一步骤已保存到 {output_file}")

def load_action_ids_from_json(json_file, id_to_find):
    action_ids = set()

    try:
        # 加载 JSON 数据
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 检查给定的 id 是否存在于数据中
        if str(id_to_find) in data:
            # 将 action_ids 添加到该 id 的集合中
            action_ids.update(data[str(id_to_find)])
    
    except Exception as e:
        print(f"无法处理文件 {json_file}: {e}")
    
    return action_ids

if __name__ == "__main__":
    # NIV 数据集处理
    directory_niv = "../dataset/NIV/processed_data"  # 替换为实际的目录路径
    prefixes_niv = ["changing_tire", "coffee", "cpr", "jump_car", "repot"]  # 替换为实际的前缀列表
    output_file_niv = "../dataset/NIV/classes_to_actions.json"  # 输出文件名
    process_files_in_directory_NIV(directory_niv, prefixes_niv, output_file_niv)

    # Coin 数据集处理
    directory_coin = "../dataset/coin/full_npy"  # 替换为实际的目录路径
    output_file_coin = "../dataset/coin/classes_to_actions.json"  # 输出文件名
    process_files_in_directory_coin(directory_coin, output_file_coin)
