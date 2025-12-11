import torch
import os
from transformers import AutoModelForCausalLM

model_name = "roneneldan/TinyStories-1M"
output_file = "model_tensors_info.txt"

print(f"Loading model from {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"\nModel architecture:\n{model}\n")

with open(output_file, "w") as out:
    header = f"=== Model: {model_name} ===\n"
    print(header)
    out.write(header + "\n")
    
    # Get state dict
    state_dict = model.state_dict()
    
    total_params = 0
    for weight_name, tensor in state_dict.items():
        params = tensor.numel()
        total_params += params
        line = f"{weight_name}: {tuple(tensor.shape)} | dtype: {tensor.dtype} | params: {params:,}"
        print(line)
        out.write(line + "\n")
    
    summary = f"\nTotal parameters: {total_params:,}"
    print(summary)
    out.write(summary + "\n")

print(f"\nResults saved to {output_file}")
