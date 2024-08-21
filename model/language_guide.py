import os
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

gpu = 6


class LLaMA3_LLM(LLM):
    # 基于本地 llama3 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_name_or_path: str):
        super().__init__()
        # print("正在从本地加载模型...")
        device = torch.device(f'cuda:{gpu}')
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # print(f"模型加载到设备: {device}")
        # print("完成本地模型的加载")

    def bulid_input(self, prompt, history=[]):
        user_format = 'user\n\n{content}'
        assistant_format = 'assistant\n\n{content}'
        history.append({'role': 'user', 'content': prompt})
        prompt_str = ''
        # 拼接历史对话
        for item in history:
            if item['role'] == 'user':
                prompt_str += user_format.format(content=item['content'])
            else:
                prompt_str += assistant_format.format(content=item['content'])
        return prompt_str

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        device = torch.device(f'cuda:{gpu}')
        input_str = self.bulid_input(prompt=prompt)
        input_ids = self.tokenizer.encode(
            input_str, add_special_tokens=False, return_tensors='pt'
        ).to(device)

        # print(f"输入张量加载到设备: {input_ids.device}")
        # print(f"模型所在设备: {self.model.device}")

        outputs = self.model.generate(
            input_ids=input_ids, max_new_tokens=512, do_sample=True,
            top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=self.tokenizer.encode('')[0]
        )

        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = self.tokenizer.decode(outputs).strip().replace('', "").replace(
            'assistant\n\n', '').strip()
        return response

    @property
    def _llm_type(self) -> str:
        return "LLaMA3_LLM"


model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B',
                              cache_dir='/datasets/zhouyufan/modelscope', revision='master')


llm = LLaMA3_LLM(
    mode_name_or_path="/datasets/zhouyufan/modelscope/LLM-Research/Meta-Llama-3-8B")

response = llm("Do you know Transformer in meachine learning?")
print('----------------------------')
print(response)
print('----------------------------')
