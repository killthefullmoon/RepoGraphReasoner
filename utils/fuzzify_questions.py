#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fuzzify (obfuscate) English multi-hop QA questions using a local instruction model.
Recommended model: Llama-3-8B-Instruct or Mistral-7B-Instruct
"""

import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

INPUT_FILE = "multi_hop_qas_en_unique.jsonl"
OUTPUT_FILE = "multi_hop_qas_en_fuzzy.jsonl"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"   # or "mistralai/Mistral-7B-Instruct-v0.3"

# ---------------------------
# 1. Load model (on GPU)
# ---------------------------
print(f"Loading model {MODEL_NAME} ...")
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    model_kwargs={"load_in_4bit": True}
)

# ---------------------------
# 2. Define prompt
# ---------------------------
PROMPT_TEMPLATE = (
    "You are an expert in rewriting questions for reasoning datasets.\n"
    "Please paraphrase or obfuscate the following question so that it becomes less literal "
    "but keeps the same meaning and is still answerable using the same reasoning path.\n"
    "Rules:\n"
    "- Replace explicit names with descriptions (e.g., 'flask' → 'a lightweight web framework').\n"
    "- Keep all logical conditions equivalent.\n"
    "- Keep it concise, one sentence.\n"
    "- Output only the rewritten question.\n\n"
    "Original question:\n{q}\n\nRewritten:"
)

# ---------------------------
# 3. Generate fuzzy versions
# ---------------------------
def fuzzify_question(q_text):
    prompt = PROMPT_TEMPLATE.format(q=q_text)
    result = generator(prompt, max_new_tokens=64, temperature=0.9, top_p=0.95)[0]["generated_text"]
    # Extract only after "Rewritten:"
    if "Rewritten:" in result:
        result = result.split("Rewritten:")[-1].strip()
    return result.strip().replace("\n", " ")


fuzzy_data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Fuzzifying"):
        item = json.loads(line)
        q_text = item["question"]
        fuzzy_q = fuzzify_question(q_text)
        item["question_fuzzy"] = fuzzy_q
        fuzzy_data.append(item)

# ---------------------------
# 4. Save
# ---------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in fuzzy_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[OK] Saved {len(fuzzy_data)} fuzzified QA pairs → {OUTPUT_FILE}")
