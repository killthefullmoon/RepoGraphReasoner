# 🚀 使用示例 - Usage Examples

## 📋 基础使用

### 1. 默认配置（推荐）

```bash
# 使用默认语言过滤：python, sh, bash, console, shell, cmd, powershell
python batch_process_repos.py --max-repos 10
```

**适用场景**：大多数Python项目

---

## 🎯 按场景使用

### 2. 机器学习/深度学习项目

```bash
# 只提取Python代码（ML库通常只有Python示例）
python batch_process_repos.py --preset ml-libs --languages python --max-repos 20

# 或使用默认配置
python batch_process_repos.py --preset ml-libs --max-repos 20
```

**示例仓库**：transformers, pytorch, tensorflow

---

### 3. Web框架项目

```bash
# Python + JavaScript（前端示例）
python batch_process_repos.py --preset web-frameworks --languages python javascript --max-repos 15
```

**示例仓库**：flask, django, fastapi

---

### 4. CLI工具项目

```bash
# 包含更多Shell脚本
python batch_process_repos.py --preset cli-tools --languages python bash sh console --max-repos 10
```

**示例仓库**：yt-dlp, thefuck, sherlock

---

### 5. DevOps/基础设施项目

```bash
# 主要是Shell脚本
python batch_process_repos.py --languages bash sh shell powershell --query "stars:>1000 language:python topic:devops"
```

**示例仓库**：ansible, docker-compose工具

---

### 6. 纯Python库（无Shell依赖）

```bash
# 只提取Python
python batch_process_repos.py --languages python --preset data-tools --max-repos 15
```

**示例仓库**：pandas, numpy, scikit-learn

---

## 🔧 高级配置

### 7. 多语言项目

```bash
# Python + Go + Rust
python batch_process_repos.py --languages python go rust --query "stars:>1000 language:python"
```

---

### 8. 自定义查询 + 语言过滤

```bash
# 深度学习 + 只提取Python
python batch_process_repos.py \
    --query "stars:>2000 language:python topic:deep-learning" \
    --languages python \
    --max-repos 10
```

---

### 9. 控制处理速度

```bash
# 增加延迟避免rate limit
python batch_process_repos.py \
    --languages python bash \
    --max-repos 20 \
    --delay 3.0
```

---

### 10. 指定输出目录

```bash
# 自定义输出路径
python batch_process_repos.py \
    --languages python \
    --max-repos 10 \
    --output ./my_custom_dataset
```

---

## 📊 测试和调试

### 11. 小规模测试

```bash
# 先处理3个repo测试配置
python batch_process_repos.py --languages python --max-repos 3 --delay 2.0
```

---

### 12. 查看日志

```bash
# 启动处理
python batch_process_repos.py --max-repos 10 &

# 另一个终端查看实时日志
tail -f batch_process.log
```

---

### 13. 检查提取的代码块

```bash
# 运行后查看日志中的代码块统计
grep "提取代码块" batch_process.log
```

**输出示例**：
```
提取代码块: setup=2, docker=1, example=5
提取代码块: setup=3, docker=0, example=8
```

---

## 🎨 特殊场景

### 14. 包含配置文件的项目（如需要YAML）

```bash
# Ansible、Kubernetes等项目可能需要YAML
python batch_process_repos.py --languages python bash yaml --query "stars:>1000 topic:ansible"
```

---

### 15. 包含前端代码的全栈项目

```bash
# Python + JavaScript + TypeScript
python batch_process_repos.py \
    --languages python javascript typescript \
    --query "stars:>1000 language:python topic:fullstack"
```

---

### 16. 系统编程项目

```bash
# Python + C + Rust
python batch_process_repos.py \
    --languages python c rust \
    --query "stars:>1000 language:python topic:systems"
```

---

## 📈 实际工作流

### 17. 完整的数据集构建流程

```bash
# Step 1: 机器学习库（只Python）
python batch_process_repos.py \
    --preset ml-libs \
    --languages python \
    --max-repos 15 \
    --output ./dataset_ml \
    --delay 2.5

# Step 2: CLI工具（Python + Shell）
python batch_process_repos.py \
    --preset cli-tools \
    --languages python bash sh \
    --max-repos 10 \
    --output ./dataset_cli \
    --delay 2.5

# Step 3: Web框架（Python + JS）
python batch_process_repos.py \
    --preset web-frameworks \
    --languages python javascript \
    --max-repos 10 \
    --output ./dataset_web \
    --delay 2.5
```

---

### 18. 合并多个数据集

```bash
# 合并所有JSONL文件
cat dataset_ml/dataset.jsonl \
    dataset_cli/dataset.jsonl \
    dataset_web/dataset.jsonl \
    > combined_dataset.jsonl

# 查看统计
wc -l combined_dataset.jsonl
```

---

## 🔍 查看结果

### 19. 快速查看生成的任务

```bash
# 查看第一个repo的任务
python -c "
import json
with open('dataset/dataset.jsonl') as f:
    repo = json.loads(f.readline())
    print(f'Repo: {repo[\"repo_name\"]}')
    print(f'Tasks: {len(repo[\"tasks\"])}')
    for i, task in enumerate(repo['tasks'][:3], 1):
        print(f'\nTask {i}: {task[\"task_title\"]}')
"
```

---

### 20. 查看语言过滤效果

```bash
# 查看某个repo的README和提取的代码块
ls dataset/readmes/
ls dataset/tasks/

# 查看具体任务文件
cat dataset/tasks/huggingface_transformers_tasks.json | python -m json.tool | head -50
```

---

## 💡 最佳实践组合

### 21. 生产环境推荐配置

```bash
# 稳定、可靠的配置
python batch_process_repos.py \
    --preset ml-libs \
    --languages python bash \
    --max-repos 30 \
    --delay 2.5 \
    --output ./production_dataset
```

---

### 22. 快速原型配置

```bash
# 快速测试和迭代
python batch_process_repos.py \
    --languages python \
    --max-repos 5 \
    --delay 1.5
```

---

## 🎯 常见问题解决

### 23. 如果提取了太多无关代码块

```bash
# 使用更严格的语言过滤
python batch_process_repos.py --languages python --max-repos 10
```

---

### 24. 如果遗漏了某些代码块

```bash
# 添加更多语言标记
python batch_process_repos.py --languages python bash sh console shell --max-repos 10
```

---

### 25. 如果rate limit太严重

```bash
# 增加延迟
python batch_process_repos.py --languages python --max-repos 10 --delay 5.0
```

---

## 📝 总结

### 默认配置适用于
- ✅ 通用Python项目
- ✅ 包含安装脚本的项目
- ✅ CLI工具
- ✅ 大多数开源库

### 自定义语言适用于
- 🎯 纯Python库（`--languages python`）
- 🎯 多语言项目（`--languages python javascript go`）
- 🎯 特定类型项目（如只要Shell脚本）

### 推荐起始命令

```bash
# 最安全的起始配置
python batch_process_repos.py --languages python bash --max-repos 5 --delay 2.5
```

检查结果后再扩大规模！
