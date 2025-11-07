#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotate multi-hop QA with CoT using an LLM backend, auto-verify final answers,
and filter out incorrect chains-of-thought (CoT).

Inputs:
  - multi_hop_qas_en_unique.jsonl  (from your unique-answer generator)

Outputs:
  - cot_dataset.jsonl     (kept, correct final answer; includes CoT)
  - cot_rejected.jsonl    (filtered out: wrong or unparsable)
  - cot_stats.json        (summary stats)

Backends:
  --backend deepseek   (needs env DEEPSEEK_API_KEY)
  --backend openai     (OpenAI-compatible; use OPENAI_API_KEY and optionally OPENAI_BASE_URL)
  --model <model_name> (e.g., deepseek-chat, deepseek-reasoner, qwen, llama, etc.)
"""

import os, json, re, time, argparse, requests
from typing import Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm

INPUT_QA = "multi_hop_qas_en_unique.jsonl"
OUT_GOOD = "cot_dataset.jsonl"
OUT_BAD  = "cot_rejected.jsonl"
OUT_STATS = "cot_stats.json"

RE_FINAL = re.compile(r"<final>\s*(.+?)\s*</final>", re.IGNORECASE | re.DOTALL)
RE_REASON = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)


COT_SYSTEM_PROMPT = (
    "You are an expert code reasoning assistant. "
    "You must reason step by step over graph facts (nodes and labeled edges) "
    "to answer the question. "
    "Strictly follow the response format below."
)

COT_USER_TEMPLATE = """You are given a multi-hop question about a code knowledge graph and a set of supporting edges.

Question:
{question}

Graph support (edges):
{edges_json}

Instructions:
1) Think step by step using the edges only.
2) Output your reasoning inside <reasoning> ... </reasoning>.
3) Output the final answer as the EXACT node id string from the graph, inside <final> ... </final>.
4) The final answer must be a SINGLE node id string that exists in the graph_support answer set.

Response format (no extra text outside the tags):
<reasoning>...your step-by-step reasoning in English...</reasoning>
<final>node_type:module_or_symbol_id</final>
"""

@dataclass
class BackendConfig:
    backend: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 512
    top_p: float = 0.9
    # rate-limit/backoff
    retry: int = 3
    retry_sleep: float = 2.0


def call_llm(cfg: BackendConfig, messages: List[Dict[str, str]]) -> str:
    """
    Return assistant message content (string). Raise on fatal errors.
    Supported backends:
      - deepseek (https://api.deepseek.com/v1/chat/completions)
      - openai-compatible (OPENAI_BASE_URL or default OpenAI endpoint)
    """
    if cfg.backend == "deepseek":
        url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }
    elif cfg.backend == "openai":
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        url = f"{base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

    last_err = None
    for _ in range(cfg.retry):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            last_err = f"HTTP {r.status_code} - {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(cfg.retry_sleep)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def parse_cot(raw: str) -> Dict[str, str]:
    final_match = RE_FINAL.search(raw or "")
    reason_match = RE_REASON.search(raw or "")
    final = final_match.group(1).strip() if final_match else ""
    reasoning = reason_match.group(1).strip() if reason_match else ""
    return {"final": final, "reasoning": reasoning}


def verify_answer(sample: Dict[str, Any], pred_final: str) -> bool:
    """
    Basic verification: final answer must equal the gold node id (unique).
    Optionally, you can add stricter checks (e.g., confirm edges pattern), but
    unique-answer dataset already guarantees a single gold.
    """
    gold = sample.get("answer", [])
    if not gold:
        return False
    return pred_final in gold and len(gold) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_QA, help="Path to unique multi-hop QA jsonl")
    ap.add_argument("--backend", choices=["deepseek", "openai"], required=True)
    ap.add_argument("--model", required=True, help="Model name for the chosen backend")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    args = ap.parse_args()

    cfg = BackendConfig(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )

    kept, rejected = 0, 0
    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(OUT_GOOD, "w", encoding="utf-8") as f_ok, \
         open(OUT_BAD, "w", encoding="utf-8") as f_bad:

        lines = f_in.readlines()
        if args.limit > 0:
            lines = lines[:args.limit]

        for line in tqdm(lines, desc="Annotating"):
            if not line.strip():
                continue
            sample = json.loads(line)

            # Build prompt
            edges_json = json.dumps(sample.get("graph_support", {}).get("edges", []), ensure_ascii=False)
            user = COT_USER_TEMPLATE.format(
                question=sample["question"],
                edges_json=edges_json
            )
            messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]

            try:
                raw = call_llm(cfg, messages)
            except Exception as e:
                sample["_error"] = f"LLM error: {e}"
                f_bad.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            parsed = parse_cot(raw)
            sample["cot"] = parsed.get("reasoning", "")
            sample["final_pred"] = parsed.get("final", "")
            sample["raw_model_output"] = raw

            # sanity: must have both tags parsed
            if not sample["cot"] or not sample["final_pred"]:
                sample["_error"] = "ParseError: missing reasoning or final"
                f_bad.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            # verify correctness
            correct = verify_answer(sample, sample["final_pred"])
            sample["is_correct"] = bool(correct)

            if correct:
                f_ok.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
            else:
                f_bad.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1

    # write stats
    stats = {
        "input": args.input,
        "backend": args.backend,
        "model": args.model,
        "kept": kept,
        "rejected": rejected,
        "kept_ratio": float(kept) / max(1, kept + rejected)
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[STATS] {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
