import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

import json, pathlib, torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from modelscope import AutoModelForCausalLM, AutoTokenizer
import transformers
from tqdm import tqdm
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from collections import defaultdict
from torch import nn


base_model = "Your LLM Path"  # The base model you initially used
adapter_path = ""  # The directory where save_pretrained was just called
max_length = 512
last_k = 32  # Only calculate the loss for the last_k real tokens of each sample

# 1. First, load the base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

tokenizer = AutoTokenizer.from_pretrained(base_model, pad_token="PAD")

####################################################

if adapter_path != "":  # 和lora adapter合并
    model = PeftModel.from_pretrained(model, adapter_path)
    # 3. Merge and unload the adapter (optional, for speedup + compatibility with old code)
    model = model.merge_and_unload()  # ← Get the complete model, use it like a regular model afterwards


# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type is causal language model
    r=8,  # Rank of LoRA
    lora_alpha=32,  # Alpha value of LoRA
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA
    lora_dropout=0.05,  # Dropout probability of LoRA
    bias="none"  # Do not fine-tune the bias term
)

# Apply LoRA configuration to the model
peft_model = get_peft_model(model, lora_config) # 加载配置
peft_model.print_trainable_parameters()  # Print trainable parameters


"""Compare parameters before and after LoRA"""

trainable_params = 0  # Number of trainable parameters
all_param = 0  # Total number of parameters

# Iterate over all parameters
for _, param in peft_model.named_parameters():
    all_param += param.numel()  # Add the number of parameters to the total
    if param.requires_grad:  # If the parameter requires gradient
        trainable_params += param.numel()  # Add the number of parameters to the trainable parameters

# Print the results 现实可训练参数量占总参数量的百分比
print(f"Number of trainable parameters: {trainable_params}")
print(f"Total number of parameters: {all_param}")
print(f"Percentage of trainable parameters: {100 * trainable_params / all_param:.2f}%")

prompts = json.loads(pathlib.Path("/home/wenbin.guo/SS4LLM/FB15k-237.json").read_text(encoding='utf-8'))
tokenized_list = []
for p in tqdm(prompts, desc="tokenizing"):
    # Do not pad to max_length here (we will pad uniformly with the collator later), but do truncation to ensure length <= max_length
    t = tokenizer(p, truncation=True, max_length=max_length, return_attention_mask=True)
    input_ids = t["input_ids"]
    attn = t.get("attention_mask", [1] * len(input_ids))
    seq_len = int(sum(attn))  # Number of real tokens that are not padded

    # Construct labels: Set positions other than the last last_k real tokens to -100 (to be ignored)
    labels = [-100] * len(input_ids) # 自回归语言建模，先用-100赋值
    start = max(0, seq_len - last_k) # 只计算最后last_k个token的损失
    for i in range(start, seq_len): # 对于last_k的token进行赋值labels
        labels[i] = input_ids[i]

    t["labels"] = labels
    tokenized_list.append(t)


print(prompts[1])

mapped_qa_dataset = Dataset.from_list(tokenized_list)

# Print a sample for verification
print("Example input_ids length:", len(mapped_qa_dataset[0]["input_ids"]))
print("Example attention_mask:", sum(mapped_qa_dataset[0]["attention_mask"]))
print("Example labels (first 50):", mapped_qa_dataset[0]["labels"][:50])


# peft_model.config.use_cache = False  # Disable caching to eliminate warnings. Re-enable for inference!
from typing import List, Dict
def collate_fn(batch: List[Dict]):
    # Convert list(dict) -> dict(tensor)
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in batch]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in batch]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in batch]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }


# Define the trainer to train the model
trainer = transformers.Trainer(
    model=peft_model,  # Model，传入和lora合并之后的模型
    train_dataset=mapped_qa_dataset,  # Training dataset
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,  # Training batch size per device
        gradient_accumulation_steps=4,  # Gradient accumulation steps
        warmup_steps=100,  # Warmup steps
        max_steps=400000,  # Maximum steps
        learning_rate=1e-3,  # Learning rate
        fp16=False,
        bf16=True,  # Whether to use half-precision floating point numbers
        logging_steps=1,  # Logging steps
        output_dir='outputs',  # Output directory
    ),
    data_collator=collate_fn  # Data collator
)

trainer.train()  # Start training


"""Save the LoRA fine-tuned model locally"""
# Define the model's identifier
model_id = "BLOOM-560m-LoRA"
# Save the fine-tuned model to the specified path
peft_model.save_pretrained("Llama-3.2-LoRA-FB15k237")
tokenizer.save_pretrained("Llama-3.2-LoRA-FB15k237")
print("Save successful")