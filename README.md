# RepoGraphReasoner

A framework for code repository understanding through knowledge graph construction and multi-hop question answering, with fine-tuning capabilities using LLaMA-Factory.

## Overview

RepoGraphReasoner extracts structural information from code repositories, builds knowledge graphs representing code relationships, and generates multi-hop question-answer pairs for training language models. The framework supports fine-tuning models using LLaMA-Factory and comprehensive evaluation on various benchmarks.

## Features

- **Code Repository Indexing**: Extract imports, symbols, and call relationships from Python repositories
- **Knowledge Graph Construction**: Build structured graphs representing code dependencies and relationships
- **Multi-hop QA Generation**: Generate complex question-answer pairs requiring reasoning across multiple code components
- **Model Fine-tuning**: Fine-tune language models using LLaMA-Factory with LoRA adapters
- **Comprehensive Evaluation**: Evaluate models on DROP, WinoGrande, GSM8K, MATH-500, and MMLU-STEM benchmarks

## Project Structure

```
RepoGraphReasoner/
├── utils/                    # Core utilities
│   ├── index_repo.py        # Repository indexing
│   ├── build_code_KG.py     # Knowledge graph construction
│   ├── gen_code_qa.py       # QA pair generation
│   ├── qwen_cot_generator.py # Chain-of-thought generation
│   └── ...
├── LLaMA-Factory/           # Fine-tuning framework (integrated)
│   ├── examples/train_lora/ # Training configurations
│   └── eval_custom_checkpoint.py # Evaluation scripts
├── dataset/                 # Input repositories
├── processed_data/          # Generated outputs (graphs, QA pairs)
├── run_pipeline.sh          # End-to-end pipeline script
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd RepoGraphReasoner

# Setup conda environment (if using LLaMA-Factory)
conda create -n llama_factory python=3.10
conda activate llama_factory

# Install dependencies
cd LLaMA-Factory
pip install -e .
```

### 2. Process a Repository

Run the complete pipeline on a repository:

```bash
./run_pipeline.sh <repo-name>
```

Example:
```bash
./run_pipeline.sh flask
```

This will:
1. Index the repository (`dataset/repos/flask/`)
2. Build the knowledge graph
3. Generate multi-hop QA pairs
4. Save outputs to `processed_data/flask/`

### 3. Fine-tune a Model

Our training configuration is located at:
```
LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml
```

To start training:
```bash
cd LLaMA-Factory
conda activate llama_factory
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

### 4. Evaluate a Checkpoint

After training, evaluate your model:

```bash
cd LLaMA-Factory
python eval_custom_checkpoint.py --checkpoint saves/<model_name>/lora/sft/checkpoint-<step>
```

This will evaluate on:
- DROP
- WinoGrande
- GSM8K
- MATH-500 (direct evaluation)
- MMLU-STEM (all sub-tasks)

## Pipeline Details

### Step 1: Repository Indexing

Extracts structural information from Python code:
- Module imports
- Function and class definitions
- Function calls
- File-level metadata

```bash
python utils/index_repo.py \
  --repo-root dataset/repos/<repo-name> \
  --output-dir processed_data/<repo-name>/index
```

### Step 2: Knowledge Graph Construction

Builds a NetworkX graph representing code relationships:
- **Node types**: Module, Library, File, Function, Test, ExternalAPI
- **Edge types**: IMPORTS, DEFINES, CALLS, IN_MODULE, IN_FILE, etc.

```bash
python utils/build_code_KG.py \
  --imports processed_data/<repo-name>/index/imports.jsonl \
  --symbols processed_data/<repo-name>/index/symbols.jsonl \
  --out-prefix processed_data/<repo-name>/graph
```

### Step 3: QA Generation

Generates multi-hop question-answer pairs from the knowledge graph:

```bash
python utils/gen_code_qa.py \
  --graph processed_data/<repo-name>/graph.pkl \
  --output processed_data/<repo-name>/multi_hop_qas_en_unique.jsonl
```

## Training Configuration

The main training configuration file is:
- `LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml`

Key settings:
- Base model: `Qwen/Qwen2.5-Coder-0.5B`
- Fine-tuning method: LoRA
- Dataset: Custom graph-based QA pairs
- Output directory: `saves/<model_name>/lora/sft/`

## Evaluation

The evaluation framework supports multiple benchmarks:

1. **Standard Benchmarks**:
   - DROP (F1 score)
   - WinoGrande (accuracy)
   - GSM8K (exact match)
   - MMLU-STEM (average accuracy across all sub-tasks)

2. **MATH-500 Direct Evaluation**:
   - Direct evaluation on HuggingFaceH4/MATH-500 dataset
   - Uses math verification for answer checking

Results are saved as JSON files in the checkpoint's evaluation directory.

## Output Files

After running the pipeline, you'll find:

```
processed_data/<repo-name>/
├── index/
│   ├── imports.jsonl      # Module imports
│   └── symbols.jsonl      # Function/class definitions
├── graph.pkl              # Knowledge graph (NetworkX)
├── graph.graphml          # GraphML format (for visualization)
├── graph.json             # JSON format (for inspection)
└── multi_hop_qas_en_unique.jsonl  # Generated QA pairs
```