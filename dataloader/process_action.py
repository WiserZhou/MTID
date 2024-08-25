import numpy as np


def read_task_info(path):

    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path, 'r') as f:
        idx = f.readline()  # task_class id
        while idx != '':
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(',')  # list of steps
            next(f)
            idx = f.readline()
    return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}


def process_and_save(task_map_path, task_info_path, output_path):

    # 读取 .npy 文件中的映射
    task_step_map = np.load(task_map_path, allow_pickle=True).item()

    # 读取任务信息
    task_info = read_task_info(task_info_path)

    # 处理映射，关联 titles 和 steps
    processed_data = {}
    for task_step, index in task_step_map.items():
        task_id, step_id = task_step.split('_')
        step_id = int(step_id) - 1  # step_id 从 1 开始，所以减去 1
        title = task_info['title'].get(task_id, 'Unknown Title')
        step = task_info['steps'].get(task_id, [])[step_id] if step_id < len(
            task_info['steps'].get(task_id, [])) else 'Unknown Step'
        processed_data[index] = {'title': title, 'step': step}

    print(processed_data)

    instructions = processed_data

    # 创建一个空集合来存储步骤
    steps = set()
    # 创建一个空字典来统计每个步骤出现的次数
    step_counts = {}

    # 遍历字典中的每一个项目
    for item in instructions.values():
        step = item['step']
        # 将步骤添加到集合中
        steps.add(step)
        # 更新步骤计数
        if step in step_counts:
            step_counts[step] += 1
        else:
            step_counts[step] = 1

    # 找出重复的步骤
    repeated_steps = {step: count for step,
                      count in step_counts.items() if count > 1}

    print(repeated_steps)

    # 保存处理后的数据
    np.save(output_path, processed_data)


def load_processed_data(file_path="/data/zhaobo/zhouyufan/PDPP-Optimize/dataloader/processed_data.npy"):

    return np.load(file_path, allow_pickle=True).item()


def get_value_by_key(processed_data, key):

    return processed_data.get(key, None)


if __name__ == "__main__":
    # 定义路径
    # task_map_path = "/data/zhaobo/zhouyufan/PDPP-Optimize/dataset/crosstask/crosstask_release/actions_one_hot.npy"
    # task_info_path = "/data/zhaobo/zhouyufan/PDPP-Optimize/dataset/crosstask/crosstask_release/tasks_primary.txt"
    # output_path = "/data/zhaobo/zhouyufan/PDPP-Optimize/dataloader/processed_data.npy"

    # # 处理并保存数据
    # process_and_save(task_map_path, task_info_path, output_path)

    # # 加载并查询处理后的数据
    # processed_data = load_processed_data(output_path)

    # specific_key = 5  # 例如查询索引为 5 的数据
    # result = get_value_by_key(processed_data, specific_key)

    # if result:
    #     print(
    #         f"Key: {specific_key}, Title: {result['title']}, Step: {result['step']}")
    # else:
    #     print(f"Key {specific_key} not found.")
    data = np.load(
        '/data/zhaobo/zhouyufan/PDPP-Optimize/dataset/crosstask/crosstask_features/_2uFcneXTkc.npy', allow_pickle=True)
    print(np.array(data, dtype=np.float32))

    print('-------------------------------------------------------')

    data2 = np.load(
        '/data/zhaobo/zhouyufan/PDPP-Optimize/dataset/coin/full_npy/ArcWeld_41_0UcBldDI0RA.npy', allow_pickle=True)
    print(data2['frames_features'])
