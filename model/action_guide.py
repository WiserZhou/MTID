from ..dataloader.process_action import load_processed_data, get_value_by_key
from ..llm_instruct.ask_api import ask_model
import torch


def language_guide(target, max_traj_len):
    target_view = target.view(-1, max_traj_len)  # 256*3
    result = []
    # 加载并查询处理后的数据
    processed_data = load_processed_data()
    # 遍历第一个维度上的每个元素
    for i in range(target_view.shape[0]):  # 256
        # 获取当前元素
        element = target_view[i]
        title_fact = []
        step_fact = []

       # 遍历当前元素的第二个维度
        for j in range(element.shape[0]):  # 3

            specific_key_fact = element[j]
            result_fact = get_value_by_key(processed_data, specific_key_fact)
            title_fact.append(result_fact['title']).strip()
            step_fact.append(result_fact['step']).strip()

        new_step, is_llm = ask_model(title_fact, step_fact)

        pred_new = []

        for value_to_find in new_step:
            # 查找与给定值相对应的键
            for key, value in processed_data.items():
                if value['step'] == value_to_find:
                    pred_new.append(key)
                    break
                    # print(f"The key for value {value_to_find} is: {key}")
        result.append(pred_new)
    result.view(-1)

    return torch.tensor(result)