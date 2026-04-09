from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from collections import defaultdict

model_name = "Your LLM Path"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map={"": 0}
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def expand_vocab_size(model, new_vocab_size):
    # 1. get
    old_embed = model.model.embed_tokens
    old_lm_head = model.lm_head
    dtype = old_embed.weight.dtype

    # 2. create new layer
    new_embed = nn.Embedding(new_vocab_size, 2048).to(dtype)
    new_lm_head = nn.Linear(2048, new_vocab_size, bias=False).to(dtype)

    # 3. copy
    with torch.no_grad():
        new_embed.weight[:128256] = old_embed.weight
        new_embed.weight[128256:] = old_embed.weight.mean(dim=0, keepdim=True)
        new_lm_head.weight[:128256] = old_lm_head.weight
        new_lm_head.weight[128256:] = old_lm_head.weight.mean(dim=0, keepdim=True)

    # 4. replace
    model.model.embed_tokens = new_embed
    model.lm_head = new_lm_head

    # 5. generation device_map and dispatch
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype=dtype,
    )
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["LlamaDecoderLayer"],
    )
    model = dispatch_model(model, device_map=device_map)

    return model

def LLM_response(prompt=""):
    if prompt == "":
        prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(model_inputs)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

    return response


tokenizer.add_tokens(['[q1]', '[q2]'], special_tokens=True)



print(len(tokenizer))
model = expand_vocab_size(model, len(tokenizer))

token_str = "[9]"            # need word
token_id = tokenizer.convert_tokens_to_ids(token_str)   # tokenizer("hello")["input_ids"][1]

print(token_id)

embed_layer = model.get_input_embeddings()   # nn.Embedding(vocab_size, hidden_size)
vec = embed_layer.weight[token_id]           # shape: [hidden_size]

print(vec)
LLM_response()
LLM_response("Give me a [q2] [q2] [q2] [q2].")
