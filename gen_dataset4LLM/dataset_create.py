import json, pathlib
import numpy as np
from pathlib import Path
from tqdm import tqdm  

def get_code(data):
    res = ''
    for item in data:
        res = res + '[' + str(item) + ']'
    return res


entity = json.loads(pathlib.Path("/home/.../SS4LLM/predict/FB15k-237/entity.json").read_text(encoding='utf-8'))
print(entity[1])

relation = json.loads(pathlib.Path("/home/.../SS4LLM/predict/FB15k-237/relation.json").read_text(encoding='utf-8'))
print(relation[1])
# 加载实体、关系和量化索引
indices_path = "/home/.../SS4LLM/predict/FB15k-237/indices.npy"  # 最后只需要使用使用离散的token序列
indices = np.load(indices_path)
print(indices.shape)
print(indices[1])

file_path = Path('/home/.../SS4LLM/predict/FB15k-237/train2id.txt')
lines = file_path.read_text().splitlines()
# int
triples = [list(map(int, line.split())) for line in lines[1:]]
print(len(triples))  # 20466
print(triples[:3])

# {
#   "0": [12, 58, 90],  # entity 0 的邻居实体 ID
#   "1": [3, 8],
#   ...
# }
neighborhood = json.loads(pathlib.Path("/home/.../SS4LLM/predict/FB15k-237/one_hop.json").read_text(encoding='utf-8'))

# 为啥候选答案是一跳邻居
res = []
for item in tqdm(triples, desc="Generate prompt", unit="strip"):
    input = f"The query triplet is ({entity[item[0]]['label']}, {relation[item[2]]['label']}, ?). \nThe quantized representation of entity {entity[item[0]]['label']} is: {get_code(indices[item[0]])}\nThe answer candidates and corresponding quantized representations are as follows: \n"
    for j in neighborhood[str(item[0])]:
        input = input + entity[j]['label'] + ': ' + f'{get_code(indices[j])}' + '\n'
    input = input + 'Please generate quantized representations of the top-1 potential answer entities:'
    # input = input + 'Please generate quantized representations of the top-10 potential answer entities, ranked from highest to lowest:'
    output = get_code(indices[item[1]])

    res.append(input + '\n' + output)

# 保存结果，也就是生成微调LLM的EA量化数据集
out_file = '/home/.../SS4LLM/FB15k-237.json'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
# input = f"The query triplet is ({?}, {?}, ?). \nThe quantized representation of entity {radiotherapy} is: {}\nThe answer candidates and corresponding quantized representations are as follows: "
