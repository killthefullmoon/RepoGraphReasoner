#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a custom checkpoint on DROP, WinoGrande, GSM8K, MATH, and MMLU-STEM
Usage: python eval_custom_checkpoint.py --checkpoint saves/qwen25_graph/lora/sft/checkpoint-20000
"""

import os
import sys
import json
import argparse
from datetime import datetime

try:
    from lm_eval import simple_evaluate
except ImportError:
    print("Installing lm-evaluation-harness...")
    os.system(f"{sys.executable} -m pip install 'lm-eval[api]>=0.4.0'")
    from lm_eval import simple_evaluate

import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import math verification
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


def extract_answer(text):
    """Extract the final answer from generated text"""
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
    
    answer_pattern = r'(?:the answer is|answer:|final answer:)\s*([^\n\.]+)'
    matches = re.findall(answer_pattern, text.lower())
    if matches:
        return matches[-1].strip()
    
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""


def normalize_answer(answer):
    """Normalize mathematical answer for comparison"""
    answer = answer.strip()
    answer = answer.replace('$', '')
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    return answer


def evaluate_math500_direct(model, tokenizer, checkpoint_path, base_model_path, output_dir, checkpoint_name):
    """Evaluate on MATH-500 dataset directly (reuses already loaded model)"""
    
    print("📥 Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"✅ Dataset loaded: {len(dataset)} problems")
    print()
    
    print(f"✅ Using already loaded model on device: {model.device}")
    print()
    
    print("⏳ Starting evaluation on 500 problems...")
    print("   This will take approximately 15-20 minutes")
    print()
    
    results = []
    correct_exact = 0
    correct_verify = 0
    total = 0
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating MATH-500")):
        problem = example['problem']
        reference_answer = example['answer']
        
        prompt = f"""Solve the following math problem. Show your work and put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_answer = extract_answer(generated_text)
        
        pred_norm = normalize_answer(predicted_answer)
        ref_norm = normalize_answer(reference_answer)
        exact_match = (pred_norm == ref_norm)
        
        math_verify_match = False
        if MATH_VERIFY_AVAILABLE:
            try:
                math_verify_match = verify(predicted_answer, reference_answer)
            except:
                math_verify_match = exact_match
        
        total += 1
        if exact_match:
            correct_exact += 1
        if math_verify_match or (not MATH_VERIFY_AVAILABLE and exact_match):
            correct_verify += 1
        
        results.append({
            "problem": problem,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "generated_text": generated_text,
            "exact_match": exact_match,
            "math_verify": math_verify_match if MATH_VERIFY_AVAILABLE else None,
        })
    
    accuracy_exact = correct_exact / total if total > 0 else 0
    accuracy_verify = correct_verify / total if total > 0 else 0
    
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
    
    # Save MATH-500 results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    math500_file = os.path.join(output_dir, f"math500_direct_{checkpoint_name}_{timestamp}.json")
    
    output_data = {
        "checkpoint": checkpoint_path,
        "dataset": "HuggingFaceH4/MATH-500",
        "base_model": base_model_path,
        "total_problems": total,
        "metrics": {
            "exact_match_accuracy": accuracy_exact,
            "exact_match_correct": correct_exact,
            "math_verify_accuracy": accuracy_verify if MATH_VERIFY_AVAILABLE else None,
            "math_verify_correct": correct_verify if MATH_VERIFY_AVAILABLE else None,
        },
        "per_problem_results": results,
    }
    
    with open(math500_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ MATH-500 results saved to: {math500_file}")
    print("=" * 80)
    
    return output_data


def _minerva_math_available():
    """Check whether optional Minerva Math deps are installed."""
    try:
        from importlib.metadata import version
        import sympy  # noqa: F401
        import math_verify  # noqa: F401

        antlr_version = version("antlr4-python3-runtime")
        if not antlr_version.startswith("4.11"):
            return False, f"antlr4-python3-runtime==4.11 required (found {antlr_version})"
        return True, None
    except Exception as exc:  # pragma: no cover - best effort guard
        return False, str(exc)


def evaluate_checkpoint(checkpoint_path, base_model_path="Qwen/Qwen2.5-Coder-0.5B", output_dir=None):
    """
    Evaluate a specific checkpoint
    
    Args:
        checkpoint_path: Path to LoRA checkpoint directory
        base_model_path: Path to base model
        output_dir: Output directory (auto-generated if None)
    """
    
    # Auto-generate output directory if not provided
    if output_dir is None:
        checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
        parent_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        model_name = os.path.basename(parent_dir)
        output_dir = f"/home/sour/LLaMA-Factory/saves/{model_name}/eval_results_{checkpoint_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"Evaluation on Custom Checkpoint")
    print("=" * 80)
    print(f"\n📂 Base Model: {base_model_path}")
    print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"📂 Output: {output_dir}")
    print()
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
        print(f"❌ Error: Invalid checkpoint directory (no adapter_model.safetensors found)")
        sys.exit(1)
    
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
    
    print("⏳ Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    model = model.merge_and_unload()
    
    print(f"✅ Model loaded on device: {model.device}")
    
    # Define tasks
    tasks = [
        "drop",
        "winogrande",
        "gsm8k",
    ]

    # Skip minerva_math to avoid antlr4 version conflict (we have MATH-500 direct evaluation)
    print("ℹ️  Skipping Minerva Math task (using MATH-500 direct evaluation instead)")
    
    # Add MMLU-STEM subjects
    mmlu_stem_tasks = [
        "mmlu_abstract_algebra",
        "mmlu_college_biology",
        "mmlu_college_chemistry",
        "mmlu_college_computer_science",
        "mmlu_college_mathematics",
        "mmlu_college_physics",
        "mmlu_elementary_mathematics",
        "mmlu_high_school_biology",
        "mmlu_high_school_chemistry",
        "mmlu_high_school_computer_science",
        "mmlu_high_school_mathematics",
        "mmlu_high_school_physics",
        "mmlu_high_school_statistics",
        "mmlu_machine_learning",
    ]
    
    tasks.extend(mmlu_stem_tasks)
    
    print(f"\n📋 Evaluating on {len(tasks)} benchmark tasks...")
    print("   This may take 1-3 hours depending on your GPU...")
    
    # Save merged model temporarily
    temp_model_path = os.path.join(output_dir, "temp_merged_model")
    print(f"⏳ Saving merged model temporarily to {temp_model_path}...")
    os.makedirs(temp_model_path, exist_ok=True)
    model.save_pretrained(temp_model_path)
    tokenizer.save_pretrained(temp_model_path)
    
    # Run evaluation
    print("⏳ Starting evaluation...")
    print("⚙️  Configuration:")
    print("   - Fixed batch size: 16")
    print("   - Sampling 20% of generation tasks for faster evaluation")
    print()
    
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={temp_model_path},dtype=bfloat16,trust_remote_code=True",
        tasks=tasks,
        num_fewshot=0,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_samples=False,
        limit=0.2,  # 20% sampling
    )
    
    # Clean up temp model
    print("\n🧹 Cleaning up temporary files...")
    import shutil
    if os.path.exists(temp_model_path):
        shutil.rmtree(temp_model_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    
    if "results" in results:
        for task_name, metrics in sorted(results["results"].items()):
            print(f"\n📌 {task_name.upper()}")
            
            if "exact_match" in metrics:
                print(f"   Exact Match: {metrics['exact_match']:.2%}")
            if "acc" in metrics:
                print(f"   Accuracy: {metrics['acc']:.2%}")
            if "acc_norm" in metrics:
                print(f"   Accuracy (normalized): {metrics['acc_norm']:.2%}")
            if "f1" in metrics:
                print(f"   F1 Score: {metrics['f1']:.2%}")
            if "math_verify" in metrics:
                print(f"   Math Verify: {metrics['math_verify']:.2%}")
            
            # Show other numeric metrics
            other_metrics = {k: v for k, v in metrics.items() 
                           if isinstance(v, (int, float)) and 
                           k not in ["exact_match", "acc", "acc_norm", "f1", "math_verify"]}
            for k, v in other_metrics.items():
                if "stderr" not in k:
                    print(f"   {k}: {v:.4f}")
    
    print("\n" + "=" * 80)
    
    # Save results
    checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"results_{checkpoint_name}_{timestamp}.json")
    
    try:
        serializable_results = {
            "checkpoint": checkpoint_path,
            "base_model": base_model_path,
            "results": results.get("results", {}),
            "configs": results.get("configs", {}),
            "versions": results.get("versions", {}),
        }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Standard evaluation results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save results: {e}")
    
    print("=" * 80)
    
    # Now evaluate MATH-500 directly (reuse the loaded model)
    print("\n" + "=" * 80)
    print("📐 Starting MATH-500 Direct Evaluation")
    print("=" * 80)
    print()
    
    math500_results = evaluate_math500_direct(
        model=model,
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name
    )
    
    return results, math500_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a custom checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory (e.g., saves/qwen25_graph/lora/sft/checkpoint-20000)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Base model path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Convert relative path to absolute if needed
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join("/home/sour/LLaMA-Factory", checkpoint_path)
    
    evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        base_model_path=args.base_model,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
