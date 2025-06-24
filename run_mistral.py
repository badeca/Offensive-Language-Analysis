import time
import pandas as pd
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from datasets import load_dataset

# === CONFIGURAÇÃO DO MODELO ===
model_id = "RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(
    model_id,
    use_safetensors=True,
    device_map="auto",
    trust_remote_code=True,
    max_memory={0: "6GiB", "cpu": "12GiB"},
    disable_exllama=True
)

print("Modelo carregado com sucesso!")
