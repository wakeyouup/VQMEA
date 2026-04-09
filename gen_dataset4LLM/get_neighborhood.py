import json
from collections import defaultdict


with open('/home/.../SS4LLM/predict/FB15k-237/train2id.txt') as f:
    triples = [list(map(int, line.split())) for line in f.read().splitlines()[1:]]

one_hop = defaultdict(list)
# 构建每个实体的一跳邻居，即将tail加入head的邻居列表中，同时把train/test/valid加入，但不包含对成

for e1, e2, _ in triples:
    one_hop[e1].append(e2)   # e1 -> e2

with open('/home/.../SS4LLM/predict/FB15k-237/test2id.txt') as f:
    triples = [list(map(int, line.split())) for line in f.read().splitlines()[1:]]

for e1, e2, _ in triples:
    one_hop[e1].append(e2)   # e1 -> e2

with open('/home/.../SS4LLM/predict/FB15k-237/valid2id.txt') as f:
    triples = [list(map(int, line.split())) for line in f.read().splitlines()[1:]]

for e1, e2, _ in triples:
    one_hop[e1].append(e2)   # e1 -> e2

# key -> str 转化成字符串之后保存
one_hop_str = {str(k): v for k, v in one_hop.items()}

out_file = '/home/.../SS4LLM/predict/FB15k-237/one_hop.json'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(one_hop_str, f, ensure_ascii=False, indent=2)

print(f'One hop neighbor has been written to {out_file}, involving {len (one_hop)} entities in total.')