#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified evaluation script for the exact benchmarks requested:
DROP, WinoGrande, GSM8K, MATH, and MMLU-STEM
"""

import os
import sys
import json
from datetime import datetime

# Install lm-eval if needed
try:
    from lm_eval import simple_evaluate
except ImportError:
    print("Installing lm-evaluation-harness...")
    os.system(f"{sys.executable} -m pip install 'lm-eval[api]>=0.4.0'")
    from lm_eval import simple_evaluate

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("=" * 80)
    print("Evaluation on DROP, WinoGrande, GSM8K, MATH, and MMLU-STEM")
    print("=" * 80)
    
    # Configuration
    base_model_path = "Qwen/Qwen2.5-Coder-0.5B"
    adapter_path = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/lora/sft/checkpoint-6000"
    output_dir = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/eval_results_6000"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Base Model: {base_model_path}")
    print(f"📂 Adapter: checkpoint-6000")
    print(f"📂 Output: {output_dir}")
    
    # Load model and tokenizer
    print("\n⏳ Loading model...")
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
    
    # Load and merge LoRA
    print("⏳ Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model = model.merge_and_unload()
    
    print(f"✅ Model loaded on device: {model.device}")
    
    # Define exact tasks
    tasks = [
        "drop",           # DROP
        "winogrande",     # WinoGrande  
        "gsm8k",          # GSM8K
        "minerva_math",   # MATH (use minerva_math for complete MATH dataset)
    ]
    
    # Add MMLU-STEM subjects individually (more stable than using the group)
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
    
    # Save merged model temporarily for evaluation
    temp_model_path = os.path.join(output_dir, "temp_merged_model")
    print(f"⏳ Saving merged model temporarily to {temp_model_path}...")
    os.makedirs(temp_model_path, exist_ok=True)
    model.save_pretrained(temp_model_path)
    tokenizer.save_pretrained(temp_model_path)
    
    # Run evaluation using the saved model path
    print("⏳ Starting evaluation...")
    print("⚙️  Configuration:")
    print("   - Fixed batch size: 16")
    print("   - Sampling 20% of generation tasks for faster evaluation")
    print("   - This will significantly speed up GSM8K and MATH")
    print()
    
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={temp_model_path},dtype=bfloat16,trust_remote_code=True",
        tasks=tasks,
        num_fewshot=0,
        batch_size=16,  # Fixed batch size instead of "auto"
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_samples=False,
        limit=0.2,  # Only evaluate 20% of each task
    )
    
    # Clean up temp model
    print("🧹 Cleaning up temporary files...")
    import shutil
    if os.path.exists(temp_model_path):
        shutil.rmtree(temp_model_path)
    
    # Print summary first (before saving, in case saving fails)
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    
    if "results" in results:
        for task_name, metrics in sorted(results["results"].items()):
            print(f"\n📌 {task_name.upper()}")
            
            # Display main metrics
            if "exact_match" in metrics:
                print(f"   Exact Match: {metrics['exact_match']:.2%}")
            if "acc" in metrics:
                print(f"   Accuracy: {metrics['acc']:.2%}")
            if "acc_norm" in metrics:
                print(f"   Accuracy (normalized): {metrics['acc_norm']:.2%}")
            if "f1" in metrics:
                print(f"   F1 Score: {metrics['f1']:.2%}")
            
            # Show any other numeric metrics
            other_metrics = {k: v for k, v in metrics.items() 
                           if isinstance(v, (int, float)) and 
                           k not in ["exact_match", "acc", "acc_norm", "f1"]}
            for k, v in other_metrics.items():
                print(f"   {k}: {v:.4f}")
    
    print("\n" + "=" * 80)
    
    # Save results (only the serializable parts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"results_{timestamp}.json")
    
    try:
        # Only save the main results dictionary, excluding non-serializable objects
        serializable_results = {
            "results": results.get("results", {}),
            "configs": results.get("configs", {}),
            "versions": results.get("versions", {}),
        }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save full results to JSON: {e}")
        print("   Results are displayed above.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

