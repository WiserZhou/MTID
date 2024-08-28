from openai import OpenAI
import os
from dataloader.process_action import load_processed_data, get_value_by_key

# 设置OpenAI API环境变量
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-Qp3T1iPXsV5jg839LqGuQ2cmMK1rHL0XuZVbKPkxvn0kSGaV"

# 初始化OpenAI客户端
client = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
)


def get_chat_response(text):
    # 创建聊天完成请求
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": text}
        ],
        model="gpt-4o",
    )

    # 返回模型的回复内容
    return chat_completion.choices[0].message.content


def most_common_element(lst):
    if not lst:  # 如果列表为空，则返回None
        return None

    from collections import Counter
    # 使用Counter统计列表中每个元素出现的次数
    count = Counter(lst)

    # 找到出现次数最多的元素及其出现次数
    max_count = max(count.values())
    # 从计数器中找到所有出现次数等于最大次数的元素
    common_elements = [element for element,
                       frequency in count.items() if frequency == max_count]

    # 如果有多个元素出现次数相同，这里返回其中一个即可
    return common_elements[0]


def to_lowercase(strings):
    return [s.lower() for s in strings]

# 示例使用
# lst = [1, 2, 3, 2, 4, 2, 5, 2, 3, 3, 3]
# print(most_common_element(lst))  # 输出：2


def valid(result, true_action_list):
    def lists_have_same_elements(list1, list2):

        # 将列表转换为集合
        set1 = set(list1)
        set2 = set(list2)

        # 比较两个集合是否相等
        return set1 == set2

    if lists_have_same_elements(result, true_action_list):
        return True
    return False


def ask_model(title_list, old_action):
    processed_data = load_processed_data()

    title = most_common_element(title_list)

    user_text = (
        f"For example, if there is a task about frying eggs with three actions, "
        f"which are \nstir fry eggs,beat eggs,put eggs in the pan\nthen you should give me the correct order, "
        f"which is: \nbeat eggs,put eggs in the pan,stir fry eggs\nThis task is related to the {title}. \n"
        f"I am now giving you a sequence of actions: {old_action}\n, but I am not sure if it is correct. "
        f"Please provide me with the sequence of actions that you think conforms to the chronological and logical order. "
        f"Do not give me actions that are not within the actions I provide to you! \n"
        f"Let's think step by step and give me the correct sequence, separated with commas in the middle, "
        f"without other punctuation or irrelevant sentences at the last line!"
    )

    print('Question:\n' + user_text)
    response = get_chat_response(user_text)
    print('Answer:\n' + response)
    response_last = get_last_line(response)
    print('last line:'+response_last)

    if ',' not in response_last:
        print("The response does not contain a comma-separated list. Retrying...")
        response = get_chat_response(user_text)
        print('Retry Answer:\n' + response)
        response_last = get_last_line(response)
        print('last line:'+response_last)

    if ',' in response_last:
        result = split_string_to_list(response_last)
        result = [strr.strip().lower() for strr in result]

        if valid(result, old_action):
            return result, True

    return old_action, False


def split_string_to_list(s):
    # 使用逗号作为分隔符分割字符串
    return s.split(',')

# # 示例调用
# title_list = ['frying eggs', 'frying eggs', 'boiling water', 'frying eggs']
# old_action = ["put eggs in the pan", "stir fry eggs", "beat eggs"]
# result = ask_model(title_list, old_action)
# print("Corrected action sequence:", result)


def get_last_line(text):
    # 使用 splitlines() 方法将字符串按行分割，然后获取最后一行
    lines = text.splitlines()
    return lines[-1] if lines else ''


if __name__ == "__main__":
    # 输入用户文本
    title = 'Make Jello Shots'
    true_action = 'pour jello powder,pour alcohol,stir mixture'
    old_action = 'pour jello powder,stir mixture,pour alcohol'
    user_text = (
        f"For example, if there is a task about frying eggs with three actions, "
        f"which are \nstir fry eggs,beat eggs,put eggs in the pan\nthen you should give me the correct order, "
        f"which is: \nbeat eggs,put eggs in the pan,stir fry eggs\nThis task is related to the {title}. \n"
        f"I am now giving you a sequence of actions: {old_action}\n, but I am not sure if it is correct. "
        f"Please provide me with the sequence of actions that you think conforms to the chronological and logical order. "
        f"Do not give me actions that are not within the actions I provide to you! \n"
        f"Let's think step by step and give me the correct sequence, separated with commas in the middle, "
        f"without other punctuation or irrelevant sentences at the last line!"
    )
    # 获取模型回复
    print('question:\n'+user_text)
    response = get_chat_response(user_text)

    # 打印回复
    print('answer:\n'+response)

    print('last line:'+get_last_line(response))

# 'Generate 70 different phrases of no more than 7 words to describe the label '{label_text}' for input into an image generation model. Each phrase should follow the format 'Turn the {label_text} [adjective],' where [adjective] is a adjective descriptive word that is highly aligned with the label and visually distinct.'
