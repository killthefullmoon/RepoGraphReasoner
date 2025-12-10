#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct evaluation on HuggingFaceH4/MATH-500 dataset
This script loads MATH-500 directly and evaluates the model without using lm-eval's built-in tasks
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import math verification
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    print("⚠️  math_verify not available, will only use exact match")
    MATH_VERIFY_AVAILABLE = False


def extract_answer(text):
    """Extract the final answer from generated text"""
    # Look for answer in \boxed{} format
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
    
    # Look for "The answer is" pattern
    answer_pattern = r'(?:the answer is|answer:|final answer:)\s*([^\n\.]+)'
    matches = re.findall(answer_pattern, text.lower())
    if matches:
        return matches[-1].strip()
    
    # Return last line as fallback
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""


def normalize_answer(answer):
    """Normalize mathematical answer for comparison"""
    # Remove whitespace
    answer = answer.strip()
    # Remove dollar signs
    answer = answer.replace('$', '')
    # Remove \text{} wrappers
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    return answer


def evaluate_math500(model_type="finetuned", checkpoint_path=None, base_model_path="Qwen/Qwen2.5-Coder-0.5B"):
    """
    Evaluate on MATH-500 dataset directly
    """
    
    print("=" * 80)
    print(f"Direct MATH-500 Evaluation - {model_type.upper()} Model")
    print("Using HuggingFaceH4/MATH-500 dataset")
    print("=" * 80)
    
    # Setup paths
    if model_type == "finetuned":
        if checkpoint_path is None:
            checkpoint_path = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/lora/sft/checkpoint-6000"
        output_dir = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/eval_results_6000"
        output_prefix = "math500_direct_finetuned"
    else:
        output_dir = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/eval_results_base_model"
        output_prefix = "math500_direct_base"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Base Model: {base_model_path}")
    if model_type == "finetuned":
        print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"📂 Output: {output_dir}")
    print(f"📊 Dataset: HuggingFaceH4/MATH-500 (exactly 500 problems)")
    print()
    
    # Load MATH-500 dataset
    print("📥 Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"✅ Dataset loaded: {len(dataset)} problems")
    print()
    
    # Load model
    print("⏳ Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if model_type == "finetuned":
        print("⏳ Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        model = model.merge_and_unload()
    
    print(f"✅ Model loaded on device: {model.device}")
    print()
    
    # Evaluate
    print("⏳ Starting evaluation on 500 problems...")
    print("   This will take approximately 15-20 minutes")
    print()
    
    results = []
    correct_exact = 0
    correct_verify = 0
    total = 0
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        problem = example['problem']
        reference_answer = example['answer']
        
        # Construct prompt
        prompt = f"""Solve the following math problem. Show your work and put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,  # Greedy decoding
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract and evaluate answer
        predicted_answer = extract_answer(generated_text)
        
        # Exact match (after normalization)
        pred_norm = normalize_answer(predicted_answer)
        ref_norm = normalize_answer(reference_answer)
        exact_match = (pred_norm == ref_norm)
        
        # Math verification (if available)
        math_verify_match = False
        if MATH_VERIFY_AVAILABLE:
            try:
                math_verify_match = verify(predicted_answer, reference_answer)
            except:
                math_verify_match = exact_match  # Fallback to exact match
        
        # Update counts
        total += 1
        if exact_match:
            correct_exact += 1
        if math_verify_match or (not MATH_VERIFY_AVAILABLE and exact_match):
            correct_verify += 1
        
        # Store result
        results.append({
            "problem": problem,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "generated_text": generated_text,
            "exact_match": exact_match,
            "math_verify": math_verify_match if MATH_VERIFY_AVAILABLE else None,
        })
    
    # Calculate metrics
    accuracy_exact = correct_exact / total if total > 0 else 0
    accuracy_verify = correct_verify / total if total > 0 else 0
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 MATH-500 DIRECT EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"📌 Total problems: {total}")
    print(f"   Exact Match: {correct_exact}/{total} = {accuracy_exact:.2%}")
    if MATH_VERIFY_AVAILABLE:
        print(f"   Math Verify: {correct_verify}/{total} = {accuracy_verify:.2%}")
    print()
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"{output_prefix}_{timestamp}.json")
    
    output_data = {
        "model_type": model_type,
        "dataset": "HuggingFaceH4/MATH-500",
        "base_model": base_model_path,
        "checkpoint": checkpoint_path if model_type == "finetuned" else None,
        "total_problems": total,
        "metrics": {
            "exact_match_accuracy": accuracy_exact,
            "exact_match_correct": correct_exact,
            "math_verify_accuracy": accuracy_verify if MATH_VERIFY_AVAILABLE else None,
            "math_verify_correct": correct_verify if MATH_VERIFY_AVAILABLE else None,
        },
        "per_problem_results": results,
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {results_file}")
    print("=" * 80)
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Direct evaluation on MATH-500 dataset")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["finetuned", "base"],
        default="finetuned",
        help="Model type: 'finetuned' (with LoRA) or 'base' (unfinetuned)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/lora/sft/checkpoint-6000",
        help="Path to LoRA checkpoint (for finetuned model)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Base model path"
    )
    
    args = parser.parse_args()
    
    evaluate_math500(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint if args.model_type == "finetuned" else None,
        base_model_path=args.base_model
    )


if __name__ == "__main__":
    main()

