# GitHub 仓库自动化处理工具

自动从 GitHub 搜索高质量仓库，提取 README，生成任务描述数据集。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export GITHUB_TOKEN="your_github_token"
export OPENAI_API_KEY="your_openai_api_key"
```

### 3. 配置参数

编辑 `config.yaml` 文件，配置搜索条件、输出目录等参数：

```yaml
search:
  default_query: "stars:>1000 language:python"
  max_results: 10

output:
  base_dir: "./dataset"

code_languages:
  - python
  - sh
  - bash
```

### 4. 运行

```bash
python batch_process_repos.py
```

## 配置说明

所有配置都在 `config.yaml` 文件中：

- **search**: 搜索相关配置
  - `default_query`: GitHub 搜索查询
  - `max_results`: 要处理的有效仓库数量
  
- **processing**: 处理相关配置
  - `delay_seconds`: 请求间隔（秒）
  
- **output**: 输出相关配置
  - `base_dir`: 输出目录
  
- **code_languages**: 要提取的代码块语言列表

## 输出结构

```
dataset/
├── readmes/          # README 文件
├── tasks/            # 任务 JSON（包含 input_to_gpt 调试信息）
├── metadata/         # 元数据
├── docker_files/     # Docker 配置文件
├── dataset.jsonl     # 完整数据集
├── summary.json      # 汇总统计
└── results.json      # 处理结果
```

## 特性

- ✅ 两步验证：确保代码是真实的使用示例
- ✅ 智能分类：区分 setup、docker 和示例代码
- ✅ 语言过滤：只提取指定语言的代码块
- ✅ 调试信息：记录发送给 GPT 的输入（仅在 tasks/*.json）
- ✅ Docker 支持：自动提取 Docker 配置文件
- ✅ 结构化输出：分离 code、command、input、output