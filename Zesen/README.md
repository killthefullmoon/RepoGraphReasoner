# GitHub数据集自动化Pipeline

一个完整的自动化系统，用于从GitHub获取高质量Python项目，提取README，并使用OpenAI生成编程任务描述。

## 🎯 功能特性

### 核心功能
- ✅ **自动搜索** GitHub高质量仓库（支持自定义查询）
- ✅ **智能获取** README文件（直接API获取，无需克隆）
- ✅ **AI生成** 基于README示例的编程任务
- ✅ **结构化存储** 任务数据集（JSON/JSONL格式）
- ✅ **批量处理** 支持处理大量仓库
- ✅ **完整日志** 详细的处理日志
- ✅ **汇总报告** 自动生成统计报告

### 高级特性
- 🚀 **并行处理**（可选）
- 📊 **预设查询**（ML、Web、数据科学等）
- 🔄 **错误恢复**（自动跳过失败的仓库）
- 📈 **进度追踪**（实时显示处理进度）
- 🎨 **灵活配置**（支持配置文件和命令行参数）

## 📁 文件说明

```
RepoIO/Zesen/
├── automation.py                    # 基础自动化脚本（单仓库处理）
├── github_dataset_pipeline.py       # 完整Pipeline（多仓库处理）
├── enhanced_task_generator.py       # 增强版任务生成器
├── batch_process_repos.py          # 批量处理脚本（推荐使用）
├── config.yaml                      # 配置文件
└── README.md                        # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install requests
```

### 2. 设置API密钥

```bash
# 必需：OpenAI API密钥
export OPENAI_API_KEY="your-openai-api-key"

# 可选：GitHub Token（提高rate limit）
export GITHUB_TOKEN="your-github-token"
```

### 3. 运行基础示例

```bash
# 处理50个高质量Python仓库
python batch_process_repos.py

# 处理10个机器学习库
python batch_process_repos.py --preset ml-libs --max-repos 10

# 自定义查询
python batch_process_repos.py --query "stars:>2000 language:python topic:web"
```

## 📖 详细使用说明

### 批量处理脚本（推荐）

`batch_process_repos.py` 是最完整的解决方案。

#### 基本用法

```bash
# 使用默认设置
python batch_process_repos.py

# 指定输出目录
python batch_process_repos.py --output ./my_dataset

# 设置最大仓库数
python batch_process_repos.py --max-repos 100
```

#### 使用预设查询

```bash
# 顶级Python项目（5000+ stars）
python batch_process_repos.py --preset top-python

# 机器学习库
python batch_process_repos.py --preset ml-libs

# Web框架
python batch_process_repos.py --preset web-frameworks

# 数据科学工具
python batch_process_repos.py --preset data-tools

# CLI工具
python batch_process_repos.py --preset cli-tools

# 自动化工具
python batch_process_repos.py --preset automation

# 教育项目
python batch_process_repos.py --preset educational

# 最近更新的热门项目
python batch_process_repos.py --preset recent-popular
```

#### 自定义GitHub搜索查询

```bash
# 按stars搜索
python batch_process_repos.py --query "stars:>3000 language:python"

# 按主题搜索
python batch_process_repos.py --query "language:python topic:deep-learning"

# 组合条件
python batch_process_repos.py --query "stars:>1000 language:python topic:web forks:>100"

# 按更新时间
python batch_process_repos.py --query "language:python pushed:>2024-06-01 stars:>500"
```

#### 高级选项

```bash
# 调整请求间隔（避免rate limit）
python batch_process_repos.py --delay 3.0

# 提供API密钥（如果不想用环境变量）
python batch_process_repos.py --openai-key sk-xxx --github-token ghp-xxx
```

### 单仓库处理脚本

`automation.py` 用于处理单个仓库。

```bash
# 处理当前目录
python automation.py

# 处理指定路径
python automation.py /path/to/repo

# 指定API密钥
python automation.py --api-key your-api-key
```

### Pipeline脚本（高级）

`github_dataset_pipeline.py` 提供更多自定义选项。

```bash
# 克隆完整仓库（而不只是README）
python github_dataset_pipeline.py --clone

# 并行处理（更快但可能触发rate limit）
python github_dataset_pipeline.py --parallel
```

## 📊 输出结构

运行后会生成以下目录结构：

```
dataset/
├── readmes/                    # README文件
│   ├── owner_repo_README.md
│   └── ...
├── tasks/                      # 生成的任务
│   ├── owner_repo_tasks.json
│   └── ...
├── metadata/                   # 元数据
│   ├── owner_repo_meta.json
│   └── ...
├── dataset.jsonl              # 完整数据集（JSONL格式）
├── summary.json               # 汇总报告
├── results.json               # 详细结果
└── batch_process.log          # 处理日志
```

### 数据集格式（JSONL）

每行是一个仓库的数据：

```json
{
  "repo_name": "owner/repo",
  "stars": 12345,
  "language": "Python",
  "tasks": [
    {
      "task_title": "任务标题",
      "task_description": "任务描述",
      "expected_input": ["输入1", "输入2"],
      "expected_output": ["输出1", "输出2"]
    }
  ],
  "timestamp": "2025-10-27T10:30:00"
}
```

### 任务JSON格式

```json
[
  {
    "task_title": "解析日期字符串",
    "task_description": "使用parser模块解析各种格式的日期字符串",
    "expected_input": [
      "from dateutil.parser import parse",
      "parse('2024-10-27')"
    ],
    "expected_output": [
      "datetime.datetime(2024, 10, 27, 0, 0)"
    ]
  }
]
```

## 🔧 配置说明

### 环境变量

```bash
# 必需
export OPENAI_API_KEY="sk-..."

# 可选（提高GitHub API限制）
export GITHUB_TOKEN="ghp-..."
```

### GitHub Token获取

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 选择 `public_repo` 权限
4. 生成并复制token

### OpenAI API Key获取

1. 访问 https://platform.openai.com/api-keys
2. 创建新的API密钥
3. 复制密钥

## 💡 最佳实践

### 1. 控制处理速度

```bash
# OpenAI有rate limit，建议设置延迟
python batch_process_repos.py --delay 3.0 --max-repos 20
```

### 2. 从小规模开始

```bash
# 先处理少量仓库测试
python batch_process_repos.py --max-repos 5
```

### 3. 使用预设查询

```bash
# 预设查询已优化，推荐使用
python batch_process_repos.py --preset ml-libs --max-repos 30
```

### 4. 监控日志

```bash
# 实时查看日志
tail -f batch_process.log
```

### 5. 检查汇总报告

```bash
# 处理完成后查看汇总
cat dataset/summary.json | python -m json.tool
```

## 📈 性能优化

### Rate Limit建议

- **GitHub API**: 
  - 无token: 60请求/小时
  - 有token: 5000请求/小时
  
- **OpenAI API**:
  - gpt-4o-mini: 500请求/分钟
  - 建议延迟: 2-3秒/请求

### 处理速度估算

- 单个仓库: ~5-10秒
- 50个仓库: ~5-10分钟
- 100个仓库: ~10-20分钟

## 🐛 故障排除

### 问题1: GitHub API Rate Limit

**解决方案:**
```bash
# 设置GitHub Token
export GITHUB_TOKEN="your-token"

# 或使用更严格的查询减少结果数
python batch_process_repos.py --query "stars:>10000 language:python"
```

### 问题2: OpenAI API超时

**解决方案:**
```bash
# 增加请求延迟
python batch_process_repos.py --delay 5.0
```

### 问题3: README内容过长

**解决方案:**
- 脚本会自动截断过长的README（默认8000字符）
- 可在代码中调整 `max_readme_length`

### 问题4: 某些仓库失败

**解决方案:**
- 查看 `batch_process.log` 了解详细错误
- 查看 `results.json` 了解失败原因
- 脚本会自动跳过失败的仓库继续处理

## 📚 高级用法

### 1. 自定义提示词

编辑 `batch_process_repos.py` 中的 `generate_tasks_openai` 方法，修改 `prompt` 变量。

### 2. 使用不同的OpenAI模型

修改代码中的模型参数：
```python
"model": "gpt-4o",  # 更强大但更贵
"model": "gpt-4o-mini",  # 性价比高（推荐）
"model": "gpt-3.5-turbo",  # 最快最便宜
```

### 3. 并行处理

```bash
# 谨慎使用，可能触发rate limit
python github_dataset_pipeline.py --parallel
```

### 4. 处理特定主题

```bash
# 深度学习
python batch_process_repos.py --query "language:python topic:deep-learning stars:>1000"

# 计算机视觉
python batch_process_repos.py --query "language:python topic:computer-vision stars:>500"

# NLP
python batch_process_repos.py --query "language:python topic:nlp stars:>800"
```

## 📊 数据分析

### 加载数据集

```python
import json

# 加载JSONL数据集
with open('dataset/dataset.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

print(f"数据集大小: {len(dataset)}")

# 统计任务数
total_tasks = sum(len(item['tasks']) for item in dataset)
print(f"总任务数: {total_tasks}")

# 按stars排序
sorted_data = sorted(dataset, key=lambda x: x['stars'], reverse=True)
print(f"Top 5: {[d['repo_name'] for d in sorted_data[:5]]}")
```

### 分析任务类型

```python
# 提取所有任务
all_tasks = []
for item in dataset:
    for task in item['tasks']:
        all_tasks.append({
            'repo': item['repo_name'],
            'title': task['task_title'],
            'description': task['task_description']
        })

# 任务标题词云分析
from collections import Counter
titles = [t['title'] for t in all_tasks]
print(f"任务示例: {titles[:10]}")
```

## 🎓 示例工作流

### 完整示例：构建ML库任务数据集

```bash
# 1. 搜索并处理机器学习库
python batch_process_repos.py \
    --preset ml-libs \
    --max-repos 30 \
    --output ./ml_dataset \
    --delay 2.5

# 2. 查看汇总
cat ml_dataset/summary.json | python -m json.tool

# 3. 检查生成的任务
ls ml_dataset/tasks/

# 4. 加载数据集进行分析
python -c "
import json
with open('ml_dataset/dataset.jsonl') as f:
    data = [json.loads(line) for line in f]
    
print(f'处理了 {len(data)} 个仓库')
print(f'生成了 {sum(len(d[\"tasks\"]) for d in data)} 个任务')
print('\\nTop 5 仓库:')
for d in sorted(data, key=lambda x: x['stars'], reverse=True)[:5]:
    print(f'  {d[\"repo_name\"]} - {d[\"stars\"]} stars - {len(d[\"tasks\"])} tasks')
"
```

## 🔐 安全注意事项

1. **不要泄露API密钥** - 使用环境变量，不要硬编码
2. **遵守API使用条款** - 不要过度请求
3. **数据使用合规** - 遵守仓库的许可证
4. **Rate Limit管理** - 合理设置延迟

## 📝 开发说明

### 扩展脚本

1. **自定义提示词**: 修改 `generate_tasks_openai` 方法
2. **添加新的预设查询**: 编辑 `create_preset_queries` 函数
3. **调整输出格式**: 修改 `save_results` 方法
4. **添加过滤器**: 在搜索后添加额外的过滤逻辑

### 测试

```bash
# 测试单个仓库
python automation.py ../dateutil

# 测试少量仓库
python batch_process_repos.py --max-repos 3 --delay 3.0
```

## 🎯 使用场景

1. **研究项目** - 构建编程任务数据集用于AI训练
2. **教育资源** - 收集高质量Python项目的学习任务
3. **代码分析** - 研究流行库的使用模式
4. **文档生成** - 自动生成教程和示例
5. **基准测试** - 创建代码生成模型的测试集

## 💰 成本估算

### OpenAI API成本（gpt-4o-mini）

- 输入: $0.15 / 1M tokens
- 输出: $0.60 / 1M tokens

估算（每个仓库）:
- 输入: ~2000 tokens
- 输出: ~500 tokens
- 成本: ~$0.0006 / 仓库

**100个仓库总成本: ~$0.06**

### GitHub API

- 免费层: 60请求/小时（无token）
- 认证: 5000请求/小时（有token）
- **推荐使用token**

## 🔄 工作流程图

```
1. 搜索GitHub仓库
   ↓
2. 获取README内容（通过API）
   ↓
3. 保存README到本地
   ↓
4. 提取代码示例
   ↓
5. 提交给OpenAI API
   ↓
6. 解析生成的任务
   ↓
7. 保存任务到JSON文件
   ↓
8. 追加到JSONL数据集
   ↓
9. 生成元数据
   ↓
10. 生成汇总报告
```

## 📦 输出示例

### summary.json
```json
{
  "total_repos": 50,
  "successful": 48,
  "failed": 2,
  "readme_found": 48,
  "tasks_generated": 48,
  "total_tasks": 156,
  "average_tasks_per_repo": 3.25,
  "language_distribution": {
    "Python": 50
  },
  "top_repos": [
    {
      "name": "psf/requests",
      "stars": 50000,
      "description": "HTTP library for Python"
    }
  ]
}
```

### dataset.jsonl（每行一条）
```json
{"repo_name": "psf/requests", "stars": 50000, "tasks": [...], "timestamp": "..."}
{"repo_name": "pallets/flask", "stars": 65000, "tasks": [...], "timestamp": "..."}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- GitHub API
- OpenAI API
- Python开源社区


