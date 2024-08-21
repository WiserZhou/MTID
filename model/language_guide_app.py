import requests
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数


def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 构建 chat 模版


def bulid_input(prompt, history=[]):
    # system_format = '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
    user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
    assistant_format = '<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n'
    history.append({'role': 'user', 'content': prompt})
    prompt_str = ''
    # 拼接历史对话
    for item in history:
        if item['role'] == 'user':
            prompt_str += user_format.format(content=item['content'])
        else:
            prompt_str += assistant_format.format(content=item['content'])
    return prompt_str


# # 创建FastAPI应用
# app = FastAPI()

# # 处理POST请求的端点


# @app.post("/")
def create_item(prompt: str):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型和分词器
    # json_post_raw = await request.json()  # 获取POST请求的JSON数据
    # json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    # json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    # prompt = json_post_list.get('prompt')  # 获取请求中的提示
    # history = json_post_list.get('history', [])  # 获取请求中的历史记录

    # messages = [
    #     # {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]

    # 调用模型进行对话生成
    input_str = bulid_input(prompt=prompt, history=[])
    input_ids = tokenizer.encode(
        input_str, add_special_tokens=False, return_tensors='pt').cuda()

    generated_ids = model.generate(
        input_ids=input_ids, max_new_tokens=512, do_sample=True,
        top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=tokenizer.encode('<|eot_id|>')[0]
    )
    outputs = generated_ids.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = response.strip().replace('<|eot_id|>', "").replace(
        '<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()  # 解析 chat 模版

    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    # answer = {
    #     "response": response,
    #     "status": 200,
    #     "time": time
    # }
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + \
        prompt + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return response  # 返回响应


# def get_completion(prompt):
#     headers = {'Content-Type': 'application/json'}
#     data = {"prompt": prompt}
#     response = requests.post(url='http://127.0.0.1:6006',
#                              headers=headers, data=json.dumps(data))
#     return response.json()['response']


# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    model_name_or_path = '/datasets/zhouyufan/modelscope/LLM-Research/Meta-Llama-3-8B'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16).cuda(6)

    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    # uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用

    print(create_item('hello!'))
