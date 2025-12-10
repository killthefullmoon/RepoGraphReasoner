#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare evaluation results between base model, coder model, and graph model
Supports all evaluation types: standard benchmarks, MATH-500 direct, etc.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def load_latest_results(results_dir, patterns=None):
    """
    Load the latest results file(s) from a directory
    Returns dict of {result_type: (data, filepath)}
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return {}
    
    if patterns is None:
        patterns = ["results*.json", "math500*.json"]
    
    all_results = {}
    
    for pattern in patterns:
        json_files = list(results_dir.glob(pattern))
        if json_files:
            latest_file = max(json_files, key=os.path.getmtime)
            with open(latest_file, 'r') as f:
                all_results[pattern] = (json.load(f), latest_file)
    
    return all_results


def extract_metrics_from_results(results_data):
    """Extract metrics from different result formats"""
    metrics = {}
    
    # Standard lm-eval format
    if "results" in results_data:
        for task_name, task_metrics in results_data["results"].items():
            # Flatten nested metrics
            for key, value in task_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"{task_name}::{key}"] = value
    
    # MATH-500 direct format
    elif "metrics" in results_data:
        for key, value in results_data["metrics"].items():
            if isinstance(value, (int, float)):
                metrics[f"MATH500_DIRECT::{key}"] = value
    
    return metrics


def get_main_metric_for_task(task_name, all_metrics_dict):
    """Determine the main metric for a task"""
    # Priority order for metrics
    metric_priority = [
        "math_verify_accuracy",
        "math_verify,none",
        "math_verify",
        "exact_match,flexible-extract",
        "exact_match",
        "acc,none",
        "acc",
        "f1,none",
        "f1",
        "em,none",
        "em",
    ]
    
    for metric_key in metric_priority:
        full_key = f"{task_name}::{metric_key}"
        if full_key in all_metrics_dict:
            return metric_key
    
    # Return first numeric metric if no priority match
    for key, value in all_metrics_dict.items():
        if key.startswith(f"{task_name}::") and isinstance(value, (int, float)):
            return key.split("::", 1)[1]
    
    return None


def get_main_benchmark_scores(metrics_dict):
    """Extract the 5 main benchmark scores: DROP, WinoGrande, GSM8K, MATH, MMLU"""
    scores = {}
    
    # DROP: use f1,none
    drop_key = "drop::f1,none"
    if drop_key in metrics_dict:
        scores["DROP"] = metrics_dict[drop_key]
    
    # WinoGrande: use acc,none
    wino_key = "winogrande::acc,none"
    if wino_key in metrics_dict:
        scores["Wino"] = metrics_dict[wino_key]
    
    # GSM8K: use exact_match,flexible-extract
    gsm8k_key = "gsm8k::exact_match,flexible-extract"
    if gsm8k_key in metrics_dict:
        scores["GSM8K"] = metrics_dict[gsm8k_key]
    
    # MATH: use math_verify_accuracy from MATH-500 direct evaluation
    math_key = "MATH500_DIRECT::math_verify_accuracy"
    if math_key in metrics_dict:
        scores["MATH"] = metrics_dict[math_key]
    
    # MMLU: calculate average of all MMLU-STEM sub-tasks
    mmlu_scores = []
    for key, value in metrics_dict.items():
        if key.startswith("mmlu_") and key.endswith("::acc,none"):
            if isinstance(value, (int, float)):
                mmlu_scores.append(value)
    if mmlu_scores:
        scores["MMLU"] = sum(mmlu_scores) / len(mmlu_scores)
    
    return scores


def compare_three_models(base_results_dict, coder_results_dict, graph_results_dict):
    """Compare results between base, coder, and graph models - simplified output"""
    
    # Extract all metrics from all models
    base_metrics = {}
    coder_metrics = {}
    graph_metrics = {}
    
    # Process base model results
    for result_type, (data, filepath) in base_results_dict.items():
        metrics = extract_metrics_from_results(data)
        base_metrics.update(metrics)
    
    # Process coder model results
    for result_type, (data, filepath) in coder_results_dict.items():
        metrics = extract_metrics_from_results(data)
        coder_metrics.update(metrics)
    
    # Process graph model results
    for result_type, (data, filepath) in graph_results_dict.items():
        metrics = extract_metrics_from_results(data)
        graph_metrics.update(metrics)
    
    # Get main benchmark scores
    base_scores = get_main_benchmark_scores(base_metrics)
    coder_scores = get_main_benchmark_scores(coder_metrics)
    graph_scores = get_main_benchmark_scores(graph_metrics)
    
    # Print comparison table
    print("=" * 80)
    print("📊 Model Comparison: Base vs Coder vs Graph")
    print("=" * 80)
    print()
    print(f"{'Benchmark':<12} {'Base':<10} {'Coder':<10} {'Graph':<10}")
    print("-" * 80)
    
    benchmarks = ["DROP", "Wino", "GSM8K", "MATH", "MMLU"]
    for bench in benchmarks:
        base_val = base_scores.get(bench, None)
        coder_val = coder_scores.get(bench, None)
        graph_val = graph_scores.get(bench, None)
        
        base_str = f"{base_val*100:.1f}" if base_val is not None else "N/A"
        coder_str = f"{coder_val*100:.1f}" if coder_val is not None else "N/A"
        graph_str = f"{graph_val*100:.1f}" if graph_val is not None else "N/A"
        
        print(f"{bench:<12} {base_str:<10} {coder_str:<10} {graph_str:<10}")
    
    print("=" * 80)
    print()


def main():
    base_dir = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/eval_results_base_model"
    coder_dir = "/home/sour/LLaMA-Factory/saves/qwen25_0.5B_coder/eval_results_6000"
    
    # Find graph model results directory
    # Try multiple possible locations
    graph_dirs = [
        "/home/sour/LLaMA-Factory/saves/lora/eval_results_checkpoint-20000",
    ]
    
    graph_dir = None
    for dir_path in graph_dirs:
        if os.path.exists(dir_path):
            graph_dir = dir_path
            break
    
    if graph_dir is None:
        # Try to find any eval_results directory in qwen25_graph
        graph_base = "/home/sour/LLaMA-Factory/saves/qwen25_graph"
        if os.path.exists(graph_base):
            for item in os.listdir(graph_base):
                if item.startswith("eval_results"):
                    graph_dir = os.path.join(graph_base, item)
                    break
        
        # Also try saves/lora directory
        if graph_dir is None:
            lora_base = "/home/sour/LLaMA-Factory/saves/lora"
            if os.path.exists(lora_base):
                for item in os.listdir(lora_base):
                    if item.startswith("eval_results"):
                        graph_dir = os.path.join(lora_base, item)
                        break
    
    # Load base model results
    base_results = load_latest_results(
        base_dir,
        patterns=["results_base*.json", "math500*.json"]
    )
    
    if not base_results:
        base_results = {}
    
    # Load coder model results
    coder_results = load_latest_results(
        coder_dir,
        patterns=["results*.json", "math500*.json"]
    )
    
    if not coder_results:
        print(f"❌ No coder model results found in {coder_dir}")
        sys.exit(1)
    
    # Load graph model results
    if graph_dir:
        graph_results = load_latest_results(
            graph_dir,
            patterns=["results*.json", "math500*.json"]
        )
    else:
        graph_results = {}
    
    if not graph_results:
        print(f"⚠️  No graph model results found")
        if not base_results:
            sys.exit(1)
    
    # Compare
    compare_three_models(base_results, coder_results, graph_results)


if __name__ == "__main__":
    main()
