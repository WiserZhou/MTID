import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B',
                              cache_dir='/datasets/zhouyufan/modelscope', revision='master')
