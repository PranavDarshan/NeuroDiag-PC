"""
inference.py
============
Inference script for OpenBioLLM-70B using 4-bit NF4 quantization across 2x A100 40GB GPUs.

Hardware : 2x NVIDIA A100-PCIE-40GB
Model    : aaditya/Llama3-OpenBioLLM-70B
Versions : transformers==4.44.2, accelerate==0.33.0, bitsandbytes==0.49.1, torch==2.6.0
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]    = "0,1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_DATASETS_OFFLINE"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = (
    "/media/rvcse22/CSERV/medlm/"
    "models--aaditya--Llama3-OpenBioLLM-70B/snapshots/"
    "7ad17ef0d2185811f731f89d20885b2f99b1e994"
)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
    max_memory={0: "36GiB", 1: "38GiB"},
    local_files_only=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print("✅ Model ready!\n")

# ── Inference ──────────────────────────────────────────────────────────────────
def ask(prompt, system="You are an expert medical AI assistant.", max_new_tokens=512, temperature=0.7, top_p=0.9):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda:1")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    generated = outputs[0][input_ids.shape[-1]:]
    print(f"\n⏱ {elapsed:.1f}s | {len(generated)} tokens | {len(generated)/elapsed:.2f} tok/s\n")
    print(tokenizer.decode(generated, skip_special_tokens=True))


# ── Run ────────────────────────────────────────────────────────────────────────
ask("Explain diabetic ketoacidosis in medical reasoning detail.")
