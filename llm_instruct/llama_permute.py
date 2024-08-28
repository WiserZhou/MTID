import transformers
import torch
from modelscope import snapshot_download

def number_to_words(n):
    if not 0 <= n < 20:
        return "Number out of range"

    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", 
             "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
             "sixteen", "seventeen", "eighteen", "nineteen"]

    return words[n]


def load_model(model,device):
    if model == 'Meta-Llama-3.1-8B-Instruct':
        snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct',
                        cache_dir='../modelscope', revision='master')
        pipeline = transformers.pipeline(
            "text-generation",
            model="../modelscope/LLM-Research/Meta-Llama-3.1-8B-Instruct/",
            model_kwargs={"torch_dtype": torch.float16},
            device=device
        )
    elif model == 'Qwen2-7B-Instruct':
        snapshot_download("qwen/Qwen2-7B-Instruct",cache_dir="../modelscope")

        pipeline = transformers.pipeline(
            "text-generation",
            model="../modelscope/qwen/Qwen2-7B-Instruct",
            model_kwargs={"torch_dtype": torch.float16},
            device=device
        ) 
    else:
        RuntimeError('Unknown model to find!')
    return pipeline

def getAct(elements):
    acts=""
    for items in elements:
        acts=acts+items+','
    return acts

def call_permute(actions=[],title='',model='Meta-Llama-3.1-8B-Instruct',device = "cuda:0" ):
    
    print(actions)
    print(title)
    
    
    pipeline = load_model(model=model,device=device)
    messages = [{"role": "system", "content": "You're an AI assistant with high intelligence.Please reply to me accurately."},]
    messages.append({"role": "user", "content": f"Imagine you are {title}. You have {number_to_words(len(actions))} actions "+ \
        f"to perform: {getAct(actions)}. To get the best result, you need to figure out the correct order of these actions."},)
    print("content:\n"+messages[1]['content'])
    outputs = pipeline(messages,max_new_tokens=1024,)
    
    respond=outputs[0]["generated_text"][-1]["content"].lower()
    print(respond)
    # 初始化一个空列表来存储动作及其索引
    actions_with_indices = []
    
    # 遍历每个动作，使用find方法找到它在回复中的位置
    for action in actions:
        # 找到动作在回复中的位置，如果动作不存在则find返回-1
        index = respond.find(action)
        # 如果动作存在，添加到列表中
        if index != -1:
            actions_with_indices.append((action, index))
    
    # 根据动作在回复中的位置对列表进行排序
    actions_with_indices.sort(key=lambda x: x[1])
    
    # 只保留动作名称，返回排序后的动作列表
    sorted_actions = [action[0] for action in actions_with_indices]
    
    return sorted_actions

# sor=call_permute(["pour jello powder","stir mixture","pour alcohol"],"make a jello-based alcoholic drink.")

# print("---------")
# print(sor)
# call_permute()