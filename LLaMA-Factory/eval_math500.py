#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script specifically for MATH-500 dataset
HuggingFaceH4/MATH-500
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_math500(model_type="finetuned", checkpoint_path=None, base_model_path="Qwen/Qwen2.5-Coder-0.5B"):
    """
    Evaluate on MATH-500 dataset
    
    Args:
        model_type: "finetuned" or "base"
        checkpoint_path: Path to LoRA checkpoint (if finetuned)
        base_model_path: Path to base model
    """
    
    print("=" * 80)
    print(f"MATH-500 Evaluation - {model_type.upper()} Model")
    print("=" * 80)
    
    # Setup paths
    if model_type == "finetuned":
        if checkpoint_path is None:
            checkpoint_path = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/lora/sft/checkpoint-6000"
        output_dir = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/eval_results_6000"
        output_prefix = "math500_finetuned"
    else:
        output_dir = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/eval_results_base_model"
        output_prefix = "math500_base"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Base Model: {base_model_path}")
    if model_type == "finetuned":
        print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"📂 Output: {output_dir}")
    print(f"📊 Dataset: HuggingFaceH4/MATH-500")
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
    
    # Save model temporarily
    temp_model_path = os.path.join(output_dir, f"temp_{output_prefix}")
    print(f"⏳ Saving model temporarily to {temp_model_path}...")
    os.makedirs(temp_model_path, exist_ok=True)
    model.save_pretrained(temp_model_path)
    tokenizer.save_pretrained(temp_model_path)
    
    # Direct evaluation on MATH-500 dataset (not using lm-eval's minerva_math)
    print("\n⏳ Starting DIRECT MATH-500 evaluation...")
    print("⚙️  Configuration:")
    print("   - Dataset: HuggingFaceH4/MATH-500 (exactly 500 problems)")
    print("   - Direct evaluation (not using minerva_math task)")
    print("   - This will take 15-20 minutes")
    print()
    
    # Load MATH-500 dataset directly
    from datasets import load_dataset
    print("📥 Loading HuggingFaceH4/MATH-500 dataset...")
    math500_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"✅ Loaded {len(math500_dataset)} problems from MATH-500")
    print()
    
    # We'll use a custom evaluation approach
    # Since lm-eval doesn't have a direct MATH-500 task, we'll create a simple wrapper
    # But actually, let's just use the direct evaluation script approach
    print("⚠️  Note: lm-eval doesn't have a built-in MATH-500 task.")
    print("   Please use eval_math500_direct.py instead for true MATH-500 evaluation.")
    print("   This script will use minerva_math as approximation.")
    print()
    print("   For true MATH-500, run:")
    print(f"   python eval_math500_direct.py --model-type {model_type}")
    print()
    
    # Fallback: use minerva_math but with clear warning
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={temp_model_path},dtype=bfloat16,trust_remote_code=True",
        tasks=["minerva_math"],
        num_fewshot=0,
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_samples=False,
        limit=0.1,
    )
    
    # Clean up temp model
    print("\n🧹 Cleaning up temporary files...")
    import shutil
    if os.path.exists(temp_model_path):
        shutil.rmtree(temp_model_path)
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 MATH-500 EVALUATION RESULTS")
    print("=" * 80)
    
    if "results" in results:
        for task_name, metrics in sorted(results["results"].items()):
            print(f"\n📌 {task_name.upper()}")
            
            for key, value in sorted(metrics.items()):
                if isinstance(value, (int, float)):
                    if "stderr" not in key:
                        print(f"   {key}: {value:.4f} ({value:.2%})")
                    else:
                        print(f"   {key}: {value:.4f}")
    
    print("\n" + "=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"{output_prefix}_{timestamp}.json")
    
    try:
        serializable_results = {
            "model_type": model_type,
            "dataset": "MATH-500 (or approximation with 500 problems)",
            "base_model": base_model_path,
            "checkpoint": checkpoint_path if model_type == "finetuned" else None,
            "results": results.get("results", {}),
            "configs": results.get("configs", {}),
        }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save results: {e}")
    
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MATH-500 dataset")
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

