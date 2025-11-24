#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate CoT annotations for multi-hop QA with Qwen using vLLM."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

_REPO_BASE = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = Path("/scratch/zmao_root/zmao98/boyuann/dataset/processed_data")
DEFAULT_REPOS_ROOT = Path("/scratch/zmao_root/zmao98/boyuann/dataset/repos")
DEFAULT_GRAPH_NAME = "graph.pkl"
DEFAULT_QA_NAME = "multi_hop_qas_en_unique.jsonl"
MAX_CODE_LINES = 400
MAX_CODE_CHARS = 4000

COT_SYSTEM_PROMPT = (
    "You are an expert code reasoning assistant. "
    "Use only the provided repository code snippets to reason about the answer. "
    "Think step by step and explain your chain-of-thought inside the reasoning tags, then respond strictly with the required XML tags."
)

COT_USER_TEMPLATE = """You are given a multi-hop question about a codebase along with relevant source code snippets extracted from the repository.

Question:
{question}

Relevant source code snippets:
{code_context}

Instructions:
1) Think step by step using ONLY the provided source code.
2) Output your reasoning inside <reasoning> ... </reasoning>.
3) Output the final answer as the EXACT node id string from the question's answer set, inside <final> ... </final>.
4) The final answer must be a SINGLE node id string that exists in the gold answer set.

Response format (do NOT output anything outside the two tags):
<reasoning>...your step-by-step reasoning in English...</reasoning>
<final>node_type:module_or_symbol_id</final>
"""

RE_REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
RE_FINAL = re.compile(r"<final>\s*(.+?)\s*</final>", re.IGNORECASE | re.DOTALL)


def parse_model_output(text: str) -> Dict[str, str]:
    reasoning = ""
    final = ""
    if text:
        reason_match = RE_REASONING.search(text)
        final_match = RE_FINAL.search(text)
        if reason_match:
            reasoning = reason_match.group(1).strip()
        if final_match:
            final = final_match.group(1).strip()
    return {"reasoning": reasoning, "final": final}


def verify_answer(sample: Dict[str, Any], pred: str) -> bool:
    gold = sample.get("answer", [])
    if not gold or not pred:
        return False
    return pred in gold and len(gold) == 1


def build_messages(
    sample: Dict[str, Any],
    code_context: str,
) -> List[Dict[str, str]]:
    user = COT_USER_TEMPLATE.format(
        question=sample.get("question", ""),
        code_context=code_context,
    )
    return [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def resolve_output_path(base: Path, suffix: str) -> Path:
    return base.with_name(base.stem + suffix + base.suffix)


def str_to_dtype(name: str) -> str:
    allowed = {"auto", "float16", "bfloat16", "float32"}
    if name not in allowed:
        raise ValueError(f"Unsupported dtype: {name}")
    return name


def load_graph(graph_path: Path):
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def get_node_data(G, node_id: str) -> Dict[str, Any]:
    if node_id in G:
        return G.nodes[node_id]
    return {}


def gather_code_nodes(G, sample: Dict[str, Any]) -> List[str]:
    nodes: List[str] = []
    seen = set()

    def add(node_id: Optional[str]):
        if node_id and node_id not in seen:
            seen.add(node_id)
            nodes.append(node_id)

    for ans in sample.get("answer", []) or []:
        add(ans)

    for node_id in sample.get("graph_support", {}).get("path", []) or []:
        data = get_node_data(G, node_id)
        if data.get("node_type") in {"Function", "Test"}:
            add(node_id)

    return nodes


def extract_code_snippet(
    repo_root: Path,
    rel_path: str,
    start_line: Optional[int],
    end_line: Optional[int],
) -> Optional[str]:
    file_path = repo_root / rel_path
    if not file_path.is_file():
        return None

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    total_lines = len(lines)
    if total_lines == 0:
        return None

    start = start_line if start_line and start_line > 0 else 1
    end = end_line if end_line and end_line > 0 else total_lines
    if end < start:
        end = start

    length = end - start + 1
    if length > MAX_CODE_LINES:
        end = start + MAX_CODE_LINES - 1

    end = min(end, total_lines)
    start = max(1, min(start, end))

    snippet_lines = lines[start - 1 : end]
    snippet = "".join(snippet_lines)
    if len(snippet) > MAX_CODE_CHARS:
        snippet = snippet[:MAX_CODE_CHARS] + "\n# ... truncated ...\n"

    header = f"File: {rel_path}:{start}-{end}"
    return f"{header}\n```python\n{snippet}\n```"


def build_code_context(G, sample: Dict[str, Any], repo_root: Path) -> str:
    if not repo_root or not repo_root.is_dir():
        return "N/A"

    sections = []
    for idx, node_id in enumerate(gather_code_nodes(G, sample), 1):
        data = get_node_data(G, node_id)
        rel_path = data.get("path")
        if not rel_path:
            continue
        snippet = extract_code_snippet(
            repo_root,
            rel_path,
            data.get("start_line"),
            data.get("end_line"),
        )
        if not snippet:
            continue
        header = (
            f"{idx}. {node_id} (type={data.get('node_type')}, module={data.get('module')}, file={rel_path})"
        )
        sections.append(f"{header}\n{snippet}")

    return "\n\n".join(sections) if sections else "N/A"


def strip_graph_edges(sample: Dict[str, Any]) -> None:
    support = sample.get("graph_support")
    if isinstance(support, dict):
        support.pop("edges", None)


def generate_vllm(llm: LLM, sampling_params: SamplingParams, prompt: str) -> str:
    """Run a single prompt through vLLM and return generated text."""
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    if not outputs:
        return ""
    first = outputs[0]
    if not first.outputs:
        return ""
    return first.outputs[0].text


def process_repository(
    qa_path: Path,
    graph_path: Path,
    repo_root: Path,
    ok_path: Path,
    reject_path: Path,
    stats_path: Path,
    tokenizer,
    llm: LLM,
    sampling_params: SamplingParams,
    limit: int,
):
    G = load_graph(graph_path)

    kept, rejected = 0, 0
    ok_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)

    with open(qa_path, "r", encoding="utf-8") as f_in, \
         open(ok_path, "w", encoding="utf-8") as f_ok, \
         open(reject_path, "w", encoding="utf-8") as f_bad:

        lines = [ln for ln in f_in if ln.strip()]
        if limit:
            lines = lines[:limit]

        for line in tqdm(lines, desc=f"Qwen vLLM CoT ({qa_path.parent.name})"):
            sample = json.loads(line)
            code_context = build_code_context(G, sample, repo_root)
            messages = build_messages(sample, code_context)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                raw = generate_vllm(llm, sampling_params, prompt)
            except Exception as exc:
                sample["_error"] = f"GenerationError: {exc}"
                strip_graph_edges(sample)
                f_bad.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            parsed = parse_model_output(raw)
            sample["cot"] = parsed.get("reasoning", "")
            sample["final_pred"] = parsed.get("final", "")
            sample["raw_model_output"] = raw

            strip_graph_edges(sample)

            if not sample["cot"] or not sample["final_pred"]:
                sample["_error"] = "ParseError: missing reasoning or final"
                strip_graph_edges(sample)
                f_bad.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            correct = verify_answer(sample, sample["final_pred"])
            sample["is_correct"] = bool(correct)

            if correct:
                f_ok.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
            else:
                strip_graph_edges(sample)
                f_bad.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected += 1

    stats = {
        "input": str(qa_path),
        "graph": str(graph_path),
        "repo": str(repo_root),
        "ok_output": str(ok_path),
        "reject_output": str(reject_path),
        "kept": kept,
        "rejected": rejected,
        "kept_ratio": float(kept) / max(1, kept + rejected),
    }
    with open(stats_path, "w", encoding="utf-8") as f_stats:
        json.dump(stats, f_stats, ensure_ascii=False, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate QA with CoT using Qwen + vLLM.")
    parser.add_argument("--input", help="Path to QA JSONL (single repository mode)")
    parser.add_argument("--graph", help="Path to graph.pkl for single repository mode")
    parser.add_argument("--repo-root", help="Path to the matching source repository for single mode")
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_ROOT),
        help="Root directory that stores processed repositories (batch mode)",
    )
    parser.add_argument(
        "--repos-dir",
        default=str(DEFAULT_REPOS_ROOT),
        help="Root directory that contains raw repositories used for code context",
    )
    parser.add_argument(
        "--qa-name",
        default=DEFAULT_QA_NAME,
        help="QA filename to look for under each processed repository",
    )
    parser.add_argument(
        "--graph-name",
        default=DEFAULT_GRAPH_NAME,
        help="Graph filename to look for under each processed repository",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model id or local path for Qwen",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype used for loading the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum new tokens to generate",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of QA samples per repo (0 = all)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM max_model_len if needed",
    )
    parser.add_argument("--output-ok", default=None, help="Accepted CoT JSONL path (single mode only)")
    parser.add_argument("--output-reject", default=None, help="Rejected samples JSONL (single mode only)")
    parser.add_argument("--output-stats", default=None, help="Stats JSON path (single mode only)")
    args = parser.parse_args()

    tasks = []

    if args.input:
        if not args.graph or not args.repo_root:
            raise ValueError("Single mode requires --graph and --repo-root")
        qa_path = Path(args.input)
        graph_path = Path(args.graph)
        repo_root = Path(args.repo_root)
        if not qa_path.is_file():
            raise FileNotFoundError(f"QA file not found: {qa_path}")
        if not graph_path.is_file():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        if not repo_root.is_dir():
            raise FileNotFoundError(f"Repo root not found: {repo_root}")

        ok_path = Path(args.output_ok) if args.output_ok else resolve_output_path(qa_path, "_cot_qwen_vllm")
        reject_path = Path(args.output_reject) if args.output_reject else resolve_output_path(qa_path, "_cot_reject")
        stats_path = (
            Path(args.output_stats)
            if args.output_stats
            else qa_path.with_name(qa_path.stem + "_cot_stats_vllm.json")
        )

        tasks.append({
            "qa_path": qa_path,
            "graph_path": graph_path,
            "repo_root": repo_root,
            "ok_path": ok_path,
            "reject_path": reject_path,
            "stats_path": stats_path,
        })
    else:
        processed_root = Path(args.processed_dir)
        repos_root = Path(args.repos_dir)
        if not processed_root.is_dir():
            raise FileNotFoundError(f"Processed directory missing: {processed_root}")
        if not repos_root.is_dir():
            raise FileNotFoundError(f"Repos directory missing: {repos_root}")

        for repo_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
            qa_path = repo_dir / args.qa_name
            graph_path = repo_dir / args.graph_name
            repo_root = repos_root / repo_dir.name
            if not (qa_path.is_file() and graph_path.is_file() and repo_root.is_dir()):
                continue
            ok_path = resolve_output_path(qa_path, "_cot_qwen_vllm")
            reject_path = resolve_output_path(qa_path, "_cot_reject")
            stats_path = qa_path.with_name(qa_path.stem + "_cot_stats_vllm.json")
            tasks.append({
                "qa_path": qa_path,
                "graph_path": graph_path,
                "repo_root": repo_root,
                "ok_path": ok_path,
                "reject_path": reject_path,
                "stats_path": stats_path,
            })

    if not tasks:
        raise RuntimeError("No repositories found to process. Check paths and filenames.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = str_to_dtype(args.dtype)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,  # pass model id so vLLM handles loading
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=dtype,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    total_kept = total_rejected = 0
    for task in tasks:
        stats = process_repository(
            qa_path=task["qa_path"],
            graph_path=task["graph_path"],
            repo_root=task["repo_root"],
            ok_path=task["ok_path"],
            reject_path=task["reject_path"],
            stats_path=task["stats_path"],
            tokenizer=tokenizer,
            llm=llm,
            sampling_params=sampling_params,
            limit=args.limit,
        )
        total_kept += stats["kept"]
        total_rejected += stats["rejected"]
        repo_name = task["qa_path"].parent.name
        print(
            f"[OK] {repo_name}: kept {stats['kept']} / rejected {stats['rejected']} → {task['ok_path']}"
        )

    if len(tasks) > 1:
        summary = {
            "processed_repos": len(tasks),
            "total_kept": total_kept,
            "total_rejected": total_rejected,
            "overall_kept_ratio": float(total_kept) / max(1, total_kept + total_rejected),
        }
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
