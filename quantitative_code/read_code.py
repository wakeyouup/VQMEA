import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

codebooks_path = "/home/.../SSQR4LLM/VQ/vq_model1/codebooks.npy"  # 保存的码本
embeddings = np.load(codebooks_path)  # shape (N, 1024)
print(embeddings.shape)
# print(embeddings)

indices_path = "/home/.../SSQR4LLM/VQ/vq_model1/indices.npy"  # 每个实体所对应的离散token序列
embeddings = np.load(indices_path)  # shape (N, 1024)
print(embeddings.shape)
# print(embeddings)

import numpy as np

codebooks = np.load(codebooks_path)   # (32, 32, 2048)
indices = np.load(indices_path)       # (N, 32)

entity_id = 0  # 假设取第0个实体
idxs = indices[entity_id]  # (32,)  每层选中的码字编号

# 取出所有量化嵌入
quant_embs = [codebooks[r, idxs[r]] for r in range(32)]  # 32个 (2048,) 向量

# 如果你想合成为最终的量化表示（Residual VQ 的做法）
quant_emb = np.sum(quant_embs, axis=0)  # (2048,)
print(len(quant_emb))
print(quant_emb)

print(indices[entity_id])