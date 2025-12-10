import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a graph-supported code reasoning model. "
    "Use the graph path and code snippets to perform multi-hop reasoning. "
    "Always output your reasoning inside <reasoning>...</reasoning> "
    "and the final answer inside <final>...</final>."
)

def normalize_reasoning(graph_path, final):
    """
    Create a clean, positive, graph-based reasoning template.
    Avoid teacher hallucinations and negative reasoning.
    """

    steps = "\n".join([f"{i+1}. Follow graph node: {node}"
                       for i, node in enumerate(graph_path)])

    reasoning = (
        "We follow the multi-hop graph path:\n"
        f"{steps}\n\n"
        f"This path shows that the target function is `{final}`.\n"
        "Therefore, the final answer is the node indicated in the last hop."
    )
    return reasoning


def convert(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    fout = output_path.open("w", encoding="utf-8")

    for line in input_path.open("r", encoding="utf-8"):
        sample = json.loads(line)

        # Basic fields
        qa_id = sample.get("id", "")
        question = sample.get("question", "")
        final = sample.get("final_pred", "")
        graph_path = sample.get("graph_support", {}).get("path", [])

        # Code context (if the original pipeline saved it)
        code_context = sample.get("code_context", None)

        # If missing, generate stable placeholder
        if not code_context or code_context == "N/A":
            code_context = (
                "```python\n"
                "# code snippet not available for this sample\n"
                "```\n"
            )

        # Build graph path block
        graph_text = "\n".join(
            f"{i+1}. {node}" for i, node in enumerate(graph_path)
        )

        # === USER MESSAGE ===
        user_msg = (
            f"Question:\n{question}\n\n"
            f"Graph Path:\n{graph_text}\n\n"
            f"Relevant Code Snippets:\n{code_context}\n"
            "Please answer with:\n"
            "<reasoning>...</reasoning>\n"
            "<final>node_id</final>"
        )

        # === ASSISTANT MESSAGE ===
        reasoning = normalize_reasoning(graph_path, final)
        assistant_msg = (
            f"<reasoning>{reasoning}</reasoning>\n"
            f"<final>{final}</final>"
        )

        # === FINAL DATA ITEM ===
        out_obj = {
            "id": qa_id,
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "user", "value": user_msg},
                {"from": "assistant", "value": assistant_msg},
            ]
        }

        fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    fout.close()
    print(f"Converted dataset saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    # python convert_to_llamafactory.py
    convert("multi_hop_qas_en_unique_cot_qwen_merged.jsonl", "train_llamafactory.jsonl")
