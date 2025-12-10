# 📐 MATH-500 单独评估指南

## 🎯 快速使用

### 评估微调模型（checkpoint-6000）

```bash
cd /home/sour/LLaMA-Factory
conda activate llama_factory
python eval_math500.py --model-type finetuned
```

### 评估基础模型（未微调）

```bash
python eval_math500.py --model-type base
```

### 指定其他 checkpoint

```bash
python eval_math500.py --model-type finetuned --checkpoint saves/path/to/checkpoint-XXXX
```

---

## 📊 关于 MATH-500

**MATH-500** 是 MATH 数据集的精选子集：
- 来源：HuggingFaceH4/MATH-500
- 题目数量：500 道题
- 覆盖：代数、几何、数论、预微积分等
- 用途：快速评估数学推理能力

---

## ⚙️ 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-type` | `finetuned` | 模型类型：`finetuned` 或 `base` |
| `--checkpoint` | `checkpoint-6000` | LoRA checkpoint 路径 |
| `--base-model` | `Qwen/Qwen2.5-Coder-0.5B` | 基础模型路径 |

---

## 📂 结果保存位置

### 微调模型
```
saves/qwen25_0.5B_coder/eval_results_6000/math500_finetuned_<timestamp>.json
```

### 基础模型
```
saves/qwen25_0.5B_coder/eval_results_base_model/math500_base_<timestamp>.json
```

---

## ⏱️ 预计时间

- **完整 500 题评估**：约 10-15 分钟（RTX 4070）
- 比完整 MATH 数据集快很多！

---

## 📈 对比 MATH-500 结果

评估完两个模型后，可以对比：

```bash
# 查看微调模型结果
cat saves/qwen25_0.5B_coder/eval_results_6000/math500_finetuned_*.json | jq .results

# 查看基础模型结果
cat saves/qwen25_0.5B_coder/eval_results_base_model/math500_base_*.json | jq .results
```

---

## 🔧 技术说明

### 评估指标

MATH-500 使用以下指标：
- **exact_match**: 精确匹配答案格式（严格）
- **math_verify**: 数学验证（推荐，更合理）

### Few-shot 设置

- 默认使用 **4-shot**（MATH 标准设置）
- 可以在脚本中修改 `num_fewshot` 参数

---

## 💡 为什么单独评估 MATH-500？

1. **更快**：500 题 vs 5000+ 题
2. **标准化**：HuggingFace 官方精选子集
3. **可复现**：社区广泛使用的基准
4. **独立运行**：不需要重新评估其他已完成的任务

---

## 🎯 示例完整工作流

```bash
# 1. 激活环境
cd /home/sour/LLaMA-Factory
conda activate llama_factory

# 2. 评估微调模型
python eval_math500.py --model-type finetuned

# 3. 评估基础模型
python eval_math500.py --model-type base

# 4. 查看和对比结果
ls -lh saves/qwen25_0.5B_coder/eval_results_6000/math500_*.json
ls -lh saves/qwen25_0.5B_coder/eval_results_base_model/math500_*.json
```

---

## 📝 结果文件格式

```json
{
  "model_type": "finetuned" or "base",
  "dataset": "MATH-500",
  "base_model": "Qwen/Qwen2.5-Coder-0.5B",
  "checkpoint": "path/to/checkpoint" (if finetuned),
  "results": {
    "task_name": {
      "exact_match": 0.1234,
      "math_verify": 0.2345,
      ...
    }
  }
}
```

---

## ⚠️ 注意事项

1. **数据集下载**：首次运行会下载 MATH-500 数据集（约 50MB）
2. **批量大小**：默认 batch_size=8，如果内存不足可以减小
3. **评估模式**：使用 4-shot（符合 MATH 标准）
4. **结果保存**：每次运行都会生成新的结果文件（不会覆盖）

---

**祝评估顺利！** 📐✨

