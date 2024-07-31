import torch


# def print_checkpoint_info(checkpoint_path):
#     # Load the checkpoint
#     checkpoint = torch.load(checkpoint_path)

#     # Print all keys in the checkpoint
#     print("Keys in the checkpoint:")
#     for key in checkpoint.keys():
#         print(key)

#     # Optionally, print detailed information for each key
#     for key, value in checkpoint.items():
#         print(f"\nKey: {key}")
#         if isinstance(value, dict):
#             print(
#                 f"Contains nested dictionary with keys: {list(value.keys())}")
#         elif isinstance(value, torch.Tensor):
#             print(f"Contains tensor of shape: {value.shape}")
#         else:
#             print(f"Contains value of type: {type(value)} and value: {value}")


# if __name__ == "__main__":
#     checkpoint_path = "/home/zhouyufan/Projects/PDPP/save_max/epoch_4layer2_0041_0.pth.tar"
#     print_checkpoint_info(checkpoint_path)

import numpy as np

# 指定.npy文件的路径
file_path = '/home/zhouyufan/Projects/PDPP/dataset/crosstask/processed_data/1373_1Gdl-A1DvpA.npy'

# 使用numpy的np.load加载文件
data = np.load(file_path,allow_pickle=True)

# 打印加载的数据
print(data)
