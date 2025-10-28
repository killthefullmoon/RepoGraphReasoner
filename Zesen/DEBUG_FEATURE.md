# 🔍 调试功能 - Debug Feature (v2.5)

## 📋 概述

新增 `input_to_gpt` 字段，用于记录发送给GPT的原始输入，方便调试和监控。

---

## ✨ 新增字段

### `input_to_gpt` 结构

```json
{
  "input_to_gpt": {
    "repo_name": "owner/repo",
    "num_code_blocks": 5,
    "total_length": 2345,
    "code_blocks": [
      "第一个代码块内容...",
      "第二个代码块内容...",
      "第三个代码块内容..."
    ]
  }
}
```

### 字段说明

- **`repo_name`**: 仓库名称
- **`num_code_blocks`**: 发送给GPT的代码块总数
- **`total_length`**: 发送给GPT的文本总长度（字符数）
- **`code_blocks`**: 前3个代码块的内容（用于快速查看）

---

## 📂 输出位置

### 1. 任务文件 (`tasks/*.json`)

```json
{
  "tasks": [...],
  "setup": {...},
  "input_to_gpt": {
    "repo_name": "huggingface/transformers",
    "num_code_blocks": 8,
    "total_length": 3456,
    "code_blocks": [
      "from transformers import pipeline\n...",
      "pipeline = pipeline('text-generation')\n...",
      "..."
    ]
  }
}
```

### 2. 数据集文件 (`dataset.jsonl`)

```json
{
  "repo_name": "huggingface/transformers",
  "tasks": [...],
  "setup": {...},
  "input_to_gpt": {
    "repo_name": "huggingface/transformers",
    "num_code_blocks": 8,
    "total_length": 3456,
    "code_blocks": [...]
  },
  "timestamp": "..."
}
```

---

## 🎯 使用场景

### 1. 调试GPT输出质量

```python
import json

# 读取任务文件
with open('dataset/tasks/huggingface_transformers_tasks.json') as f:
    data = json.load(f)

# 查看发送给GPT的内容
gpt_input = data['input_to_gpt']
print(f"发送了 {gpt_input['num_code_blocks']} 个代码块")
print(f"总长度: {gpt_input['total_length']} 字符")
print("\n前3个代码块:")
for i, block in enumerate(gpt_input['code_blocks'], 1):
    print(f"\n代码块 {i}:")
    print(block[:200] + "..." if len(block) > 200 else block)
```

### 2. 分析代码块数量与任务质量的关系

```python
import json

# 读取dataset.jsonl
with open('dataset/dataset.jsonl') as f:
    repos = [json.loads(line) for line in f]

# 分析
for repo in repos:
    num_blocks = repo['input_to_gpt']['num_code_blocks']
    num_tasks = len(repo['tasks'])
    print(f"{repo['repo_name']}: {num_blocks} blocks → {num_tasks} tasks")
```

### 3. 检查是否发送了过多/过少的代码

```python
import json

# 读取数据集
with open('dataset/dataset.jsonl') as f:
    repos = [json.loads(line) for line in f]

# 检查异常情况
for repo in repos:
    gpt_input = repo['input_to_gpt']
    
    # 代码块太少
    if gpt_input['num_code_blocks'] < 2:
        print(f"⚠️ {repo['repo_name']}: 只有 {gpt_input['num_code_blocks']} 个代码块")
    
    # 内容太短
    if gpt_input['total_length'] < 500:
        print(f"⚠️ {repo['repo_name']}: 内容太短 ({gpt_input['total_length']} 字符)")
    
    # 内容被截断
    if gpt_input['total_length'] >= 8000:
        print(f"⚠️ {repo['repo_name']}: 内容可能被截断")
```

### 4. 对比输入和输出

```python
import json

with open('dataset/dataset.jsonl') as f:
    repo = json.loads(f.readline())

gpt_input = repo['input_to_gpt']
tasks = repo['tasks']

print(f"仓库: {repo['repo_name']}")
print(f"\n输入:")
print(f"  - 代码块数: {gpt_input['num_code_blocks']}")
print(f"  - 总长度: {gpt_input['total_length']}")

print(f"\n输出:")
print(f"  - 任务数: {len(tasks)}")
for i, task in enumerate(tasks, 1):
    print(f"  - Task {i}: {task['task_title']}")
```

---

## 📊 实际示例

### 示例1: 查看具体发送的代码

```bash
# 查看某个repo发送给GPT的代码
python -c "
import json
with open('dataset/tasks/fastapi_fastapi_tasks.json') as f:
    data = json.load(f)
    gpt_input = data['input_to_gpt']
    print(f'Repo: {gpt_input[\"repo_name\"]}')
    print(f'代码块数: {gpt_input[\"num_code_blocks\"]}')
    print(f'总长度: {gpt_input[\"total_length\"]} 字符')
    print('\n前3个代码块:')
    for i, block in enumerate(gpt_input['code_blocks'], 1):
        print(f'\n=== 代码块 {i} ===')
        print(block)
"
```

**输出示例**:
```
Repo: fastapi/fastapi
代码块数: 5
总长度: 2345 字符

前3个代码块:

=== 代码块 1 ===
@app.get("/")
def read_root():
    return {"Hello": "World"}

=== 代码块 2 ===
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

=== 代码块 3 ===
...
```

### 示例2: 统计所有repo的代码块数量

```bash
python -c "
import json

with open('dataset/dataset.jsonl') as f:
    repos = [json.loads(line) for line in f]

print('代码块数量统计:')
print('=' * 50)
for repo in repos:
    gpt_input = repo['input_to_gpt']
    print(f'{repo[\"repo_name\"]:40s} {gpt_input[\"num_code_blocks\"]:3d} blocks')
"
```

**输出示例**:
```
代码块数量统计:
==================================================
huggingface/transformers                   8 blocks
fastapi/fastapi                            5 blocks
yt-dlp/yt-dlp                             12 blocks
...
```

### 示例3: 查找内容被截断的repo

```bash
python -c "
import json

with open('dataset/dataset.jsonl') as f:
    repos = [json.loads(line) for line in f]

print('内容被截断的仓库:')
for repo in repos:
    gpt_input = repo['input_to_gpt']
    if gpt_input['total_length'] >= 7500:  # 接近8000的限制
        print(f'{repo[\"repo_name\"]}: {gpt_input[\"total_length\"]} 字符')
"
```

---

## 💡 调试技巧

### 1. 验证语言过滤是否生效

```python
import json

# 检查提取的代码块是否只包含预期的语言
with open('dataset/tasks/pytorch_pytorch_tasks.json') as f:
    data = json.load(f)
    
    for block in data['input_to_gpt']['code_blocks']:
        # 检查是否包含非Python代码
        if '{' in block and '"' in block and block.strip().startswith('{'):
            print("⚠️ 可能包含JSON代码块！")
        elif 'version:' in block and 'services:' in block:
            print("⚠️ 可能包含YAML代码块！")
```

### 2. 检查代码块质量

```python
import json

with open('dataset/dataset.jsonl') as f:
    repos = [json.loads(line) for line in f]

for repo in repos:
    gpt_input = repo['input_to_gpt']
    
    # 检查代码块是否太短（可能是无效的）
    for i, block in enumerate(gpt_input['code_blocks']):
        if len(block.strip()) < 50:
            print(f"⚠️ {repo['repo_name']}: 代码块 {i+1} 太短 ({len(block)} 字符)")
```

### 3. 对比不同版本的输入

如果你修改了代码块提取逻辑，可以对比前后的差异：

```python
import json

# 旧版本的输出
with open('dataset_old/tasks/repo_tasks.json') as f:
    old_data = json.load(f)

# 新版本的输出
with open('dataset_new/tasks/repo_tasks.json') as f:
    new_data = json.load(f)

old_input = old_data['input_to_gpt']
new_input = new_data['input_to_gpt']

print(f"代码块数: {old_input['num_code_blocks']} → {new_input['num_code_blocks']}")
print(f"总长度: {old_input['total_length']} → {new_input['total_length']}")
```

---

## 🔧 自定义调试信息

如果你想要记录更多信息，可以修改代码中的 `gpt_input` 字典：

```python
# 在 batch_process_repos.py 中
gpt_input = {
    "repo_name": repo_name,
    "num_code_blocks": len(example_code),
    "total_length": len(example_text),
    "code_blocks": example_code[:3],  # 可以调整记录的数量
    
    # 可选：添加更多调试信息
    "language_filter": self.code_block_languages,  # 使用的语言过滤器
    "was_truncated": len(example_text) >= max_length,  # 是否被截断
    "timestamp": datetime.now().isoformat()  # 处理时间
}
```

---

## 📈 版本历史

- **v2.0** - 代码块分类 + Setup提取
- **v2.1** - While循环搜索
- **v2.2** - 两步验证机制
- **v2.3** - 增强数据结构
- **v2.4** - 语言过滤
- **v2.5** - **调试功能** ✨ 当前版本

---

## 🎯 最佳实践

1. **定期检查 `input_to_gpt`**
   - 确保发送给GPT的代码质量高
   - 验证语言过滤是否正确工作

2. **分析代码块数量与任务质量的关系**
   - 代码块太少 → 可能生成的任务不够丰富
   - 代码块太多 → 可能被截断或包含重复内容

3. **使用 `code_blocks` 快速预览**
   - 不需要重新读取README
   - 快速了解发送给GPT的内容

4. **监控内容长度**
   - 接近8000字符的限制时可能需要调整策略
   - 考虑提取更有代表性的代码块

---

## 🚀 立即使用

```bash
# 运行脚本
python batch_process_repos.py --max-repos 5

# 查看生成的调试信息
cat dataset/tasks/huggingface_transformers_tasks.json | python -m json.tool | grep -A 10 "input_to_gpt"

# 或者使用Python脚本分析
python -c "
import json
with open('dataset/dataset.jsonl') as f:
    repo = json.loads(f.readline())
    print(json.dumps(repo['input_to_gpt'], indent=2, ensure_ascii=False))
"
```

---

**状态**: ✅ 完成  
**用途**: 调试和监控  
**输出位置**: `tasks/*.json` 和 `dataset.jsonl`
