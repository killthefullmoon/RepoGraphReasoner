#!/usr/bin/env python3
"""
批量处理GitHub仓库的完整自动化脚本
整合了仓库搜索、README获取、任务生成的完整流程
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
import base64
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_process.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RepoMetadata:
    """仓库元数据"""
    full_name: str
    stars: int
    forks: int
    language: str
    description: str
    topics: List[str]
    url: str
    created_at: str
    updated_at: str
    
    def to_dict(self):
        return asdict(self)


@dataclass  
class ProcessResult:
    """处理结果"""
    repo_name: str
    success: bool
    readme_found: bool
    tasks_generated: bool
    num_tasks: int
    error: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class BatchRepoProcessor:
    """批量仓库处理器"""
    
    def __init__(self, 
                 github_token: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 output_dir: str = "./dataset",
                 code_block_languages: Optional[List[str]] = None):
        """
        初始化处理器
        
        Args:
            github_token: GitHub API Token
            openai_api_key: OpenAI API密钥
            output_dir: 输出目录
            code_block_languages: 要提取的代码块语言列表（默认: ['python', 'sh', 'bash', 'console']）
        """
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.output_dir = Path(output_dir)
        
        # 默认语言列表
        if code_block_languages is None:
            self.code_block_languages = ['python', 'sh', 'bash', 'console', 'shell', 'cmd', 'powershell']
        else:
            self.code_block_languages = code_block_languages
        
        # 创建目录结构
        self.readmes_dir = self.output_dir / "readmes"
        self.tasks_dir = self.output_dir / "tasks" 
        self.metadata_dir = self.output_dir / "metadata"
        self.dataset_file = self.output_dir / "dataset.jsonl"
        
        for dir_path in [self.readmes_dir, self.tasks_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # API配置
        self.github_api = "https://api.github.com"
        self.openai_api = "https://api.openai.com/v1/chat/completions"
        
        self.github_headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            self.github_headers["Authorization"] = f"token {self.github_token}"
        
        logger.info(f"处理器初始化完成 - 输出: {self.output_dir}")
    
    def search_repos(self, 
                    query: str,
                    sort: str = "stars",
                    max_results: int = 100) -> List[RepoMetadata]:
        """
        搜索GitHub仓库
        
        Args:
            query: 搜索查询
            sort: 排序方式
            max_results: 最大结果数
        
        Returns:
            仓库元数据列表
        """
        logger.info(f"搜索仓库 - 查询: {query}, 最大结果: {max_results}")
        
        repos = []
        page = 1
        per_page = min(100, max_results)
        
        while len(repos) < max_results:
            params = {
                "q": query,
                "sort": sort,
                "order": "desc",
                "per_page": per_page,
                "page": page
            }
            
            try:
                response = requests.get(
                    f"{self.github_api}/search/repositories",
                    headers=self.github_headers,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                items = data.get("items", [])
                
                if not items:
                    break
                
                for item in items:
                    if len(repos) >= max_results:
                        break
                    
                    repos.append(RepoMetadata(
                        full_name=item["full_name"],
                        stars=item["stargazers_count"],
                        forks=item["forks_count"],
                        language=item.get("language", "Unknown"),
                        description=item.get("description", ""),
                        topics=item.get("topics", []),
                        url=item["html_url"],
                        created_at=item["created_at"],
                        updated_at=item["updated_at"]
                    ))
                
                if len(items) < per_page:
                    break
                
                page += 1
                time.sleep(1)  # 避免rate limit
                
            except Exception as e:
                logger.error(f"搜索仓库失败: {e}")
                break
        
        logger.info(f"找到 {len(repos)} 个仓库")
        return repos
    
    def get_readme(self, repo_full_name: str) -> tuple[Optional[str], Optional[str]]:
        """
        获取README内容
        
        Args:
            repo_full_name: 仓库全名
        
        Returns:
            (readme_content, readme_filename)
        """
        url = f"{self.github_api}/repos/{repo_full_name}/readme"
        
        try:
            response = requests.get(url, headers=self.github_headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            content = base64.b64decode(data["content"]).decode('utf-8', errors='ignore')
            filename = data["name"]
            
            logger.info(f"获取README成功: {repo_full_name}/{filename}")
            return content, filename
            
        except Exception as e:
            logger.warning(f"获取README失败 {repo_full_name}: {e}")
            return None, None
    
    def extract_code_blocks(self, readme_content: str) -> Dict[str, List[str]]:
        """
        从README中提取代码块并分类
        
        Args:
            readme_content: README内容
        
        Returns:
            分类的代码块字典
        """
        code_blocks = {
            'setup_commands': [],     # 环境设置命令
            'example_code': [],       # 示例代码
            'docker_commands': []     # Docker相关命令
        }
        
        lines = readme_content.split('\n')
        in_code_block = False
        current_block = []
        current_block_type = None
        
        # 环境设置关键词
        setup_keywords = [
            'pip install', 'conda install', 'npm install', 'yarn add',
            'apt-get', 'brew install', 'requirements.txt', 'setup.py',
            'python -m venv', 'conda create', 'poetry install',
            'git clone', 'wget', 'curl','powershell',
            'sudo apt-get install','sudo dnf install'
        ]
        
        # Docker关键词
        docker_keywords = [
            'docker build', 'docker run', 'docker-compose', 'dockerfile',
            'docker pull', 'docker push', 'docker exec'
        ]
        
        for i, line in enumerate(lines):
            # 检测markdown代码块
            if line.strip().startswith('```'):
                if not in_code_block:
                    # 开始代码块 - 检查语言标记
                    in_code_block = True
                    current_block = []
                    current_block_type = None
                    
                    # 提取语言标记（如 ```python, ```bash）
                    lang_marker = line.strip()[3:].strip().lower()
                    
                    # 检查是否是我们关心的语言
                    if lang_marker:
                        # 检查是否匹配任何配置的语言
                        for lang in self.code_block_languages:
                            if lang_marker.startswith(lang):
                                current_block_type = 'potential_code'
                                break
                        
                        # 如果不是我们关心的语言，跳过这个代码块
                        if current_block_type is None:
                            in_code_block = False
                else:
                    # 结束代码块 - 分类
                    if current_block and current_block_type == 'potential_code':
                        block_text = '\n'.join(current_block)
                        block_lower = block_text.lower()
                        
                        # 判断代码块类型
                        if any(kw in block_lower for kw in docker_keywords):
                            code_blocks['docker_commands'].append(block_text)
                        elif any(kw in block_lower for kw in setup_keywords):
                            code_blocks['setup_commands'].append(block_text)
                        else:
                            # 可能是示例代码
                            code_blocks['example_code'].append(block_text)
                    
                    in_code_block = False
                    current_block = []
                    current_block_type = None
            elif in_code_block and current_block_type == 'potential_code':
                current_block.append(line)
            # 检测reStructuredText代码块或命令
            elif line.strip().startswith('$ ') or line.strip().startswith('# '):
                command = line.strip()[2:].strip()
                if any(kw in command.lower() for kw in docker_keywords):
                    code_blocks['docker_commands'].append(command)
                elif any(kw in command.lower() for kw in setup_keywords):
                    code_blocks['setup_commands'].append(command)
        
        return code_blocks
    
    def get_docker_files(self, repo_full_name: str) -> Dict[str, str]:
        """
        获取仓库中的Docker相关文件
        
        Args:
            repo_full_name: 仓库全名
        
        Returns:
            Docker文件内容字典
        """
        docker_files = {}
        docker_file_names = [
            'Dockerfile',
            'docker-compose.yml',
            'docker-compose.yaml',
            '.dockerignore'
        ]
        
        for filename in docker_file_names:
            url = f"{self.github_api}/repos/{repo_full_name}/contents/{filename}"
            
            try:
                response = requests.get(url, headers=self.github_headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    content = base64.b64decode(data["content"]).decode('utf-8', errors='ignore')
                    docker_files[filename] = content
                    logger.info(f"获取Docker文件成功: {repo_full_name}/{filename}")
            except Exception as e:
                # 文件不存在或其他错误，跳过
                continue
        
        return docker_files

    def validate_code_is_usage_example(self, readme_content: str, example_code: List[str], repo_name: str) -> bool:
        """
        第一步：使用GPT判断代码块是否是repo的usage example
        
        Args:
            readme_content: README完整内容
            example_code: 示例代码列表
            repo_name: 仓库名称
        
        Returns:
            True if 代码是usage example, False otherwise
        """
        if not self.openai_api_key:
            logger.error("未提供OpenAI API密钥")
            return False
        
        # 组合代码块
        code_samples = "\n\n---CODE BLOCK---\n\n".join(example_code)  # 只看前3个代码块
        
        prompt = f"""
你是一个代码分析专家。请判断以下代码块是否是展示该仓库/工具的使用示例。

仓库名称: {repo_name}

README内容:
{readme_content}

代码块示例:
{code_samples}

判断标准:
✅ 正面例子（返回true）:
- 代码展示如何使用这个库/工具
- 代码调用了这个仓库提供的API/功能
- 代码是用户使用该工具的示例
- 例如: import vllm; model = vllm.LLM(...)

❌ 反面例子（返回false）:
- 代码只是展示一个算法/数据结构的实现
- 仓库本身是教程/书籍，代码是教学内容而非工具使用
- 代码不涉及调用该仓库的功能
- 例如: 算法书中的排序算法实现，ML 入门教程，或者系统设计教程等

请只返回JSON格式:
{{
    "is_usage_example": true/false,
    "reason": "简短的判断理由"
}}

只返回JSON，不要其他内容。
"""
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "你是代码分析专家，总是返回有效的JSON。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.openai_api, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 清理markdown
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # 解析结果
            validation_result = json.loads(content)
            is_valid = validation_result.get("is_usage_example", False)
            reason = validation_result.get("reason", "")
            
            logger.info(f"代码验证结果: {'✓ 通过' if is_valid else '✗ 不通过'} - {reason}")
            return is_valid
            
        except Exception as e:
            logger.error(f"代码验证失败 {repo_name}: {e}")
            # 如果验证失败，保守起见返回True继续处理
            return True
    
    def generate_tasks_openai(self, example_code: List[str], repo_name: str) -> Optional[Dict]:
        """
        第二步：使用OpenAI生成任务（只基于示例代码）
        
        Args:
            example_code: 示例代码列表
            repo_name: 仓库名称
        
        Returns:
            任务数据
        """
        if not self.openai_api_key:
            logger.error("未提供OpenAI API密钥")
            return None
        
        # 检查是否有示例代码
        if not example_code:
            logger.warning(f"没有示例代码可供分析: {repo_name}")
            return None

        for i in example_code:
            print("i: ", i)
        # 组合所有示例代码
        example_text = "\n\n---\n\n".join(example_code)
        
        # 记录发送给GPT的原始输入
        gpt_input = {
            "repo_name": repo_name,
            "num_code_blocks": len(example_code),
            "total_length": len(example_text),
            "code_blocks": example_code
        }
        
        prompt = f"""
请分析以下Python库的示例代码，提取其中展示的所有功能任务。

重要说明：
1. 只分析功能性代码，不要包含环境设置相关的内容
2. 专注于代码展示的功能和用法
3. 区分代码示例和CLI命令

仓库: {repo_name}

示例代码:
{example_text}

请返回JSON格式，包含:
{{
    "tasks": [
        {{
            "task_title": "任务标题",
            "task_description": "任务描述（说明这个任务做什么，使用了什么功能）",
            "example_code": "完整的代码片段（如果是代码示例）或null（如果是CLI命令）",
            "running_command": "命令行命令（如果是CLI工具）或null（如果是代码示例）",
            "expected_input": "纯输入数据（不含代码/命令部分）",
            "expected_output": "运行结果输出"
        }}
    ]
}}

格式说明：

1. 代码示例（如transformers）：
{{
    "example_code": "from transformers import pipeline\\n\\npipeline = pipeline(task=\\"text-generation\\", model=\\"Qwen/Qwen2.5-1.5B\\")\\noutput = pipeline(\\"the secret to baking\\")\\nprint(output)",
    "running_command": null,
    "expected_input": "the secret to baking",
    "expected_output": "{{'generated_text': '...'}}"
}}

2. CLI命令（如yt-dlp）：
{{
    "example_code": null,
    "running_command": "yt-dlp -x",
    "expected_input": "https://www.youtube.com/watch?v=BaW_jenozKc",
    "expected_output": "Extracting audio..."
}}

注意：
- 只提取README中的功能性任务示例
- 不要包含安装、配置、环境设置等内容
- expected_input是纯数据（URL、文本等），不含代码或命令
- expected_output是实际运行结果
- example_code是完整可运行的代码片段
- running_command是命令行工具的命令（不含输入数据）

只返回JSON，不要其他内容。
"""
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "你是代码分析专家，总是返回有效的JSON。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.5
        }
        
        try:
            response = requests.post(self.openai_api, headers=headers, json=data, timeout=90)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 清理可能的markdown代码块
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # 解析JSON
            tasks_data = json.loads(content)
            
            # 添加input_to_gpt字段用于调试
            tasks_data['input_to_gpt'] = gpt_input
            
            return tasks_data
            
        except Exception as e:
            logger.error(f"生成任务失败 {repo_name}: {e}")
            return None
    
    def process_repo(self, repo: RepoMetadata) -> ProcessResult:
        """
        处理单个仓库
        
        Args:
            repo: 仓库元数据
        
        Returns:
            处理结果
        """
        logger.info(f"处理仓库: {repo.full_name}")
        
        # 提取仓库名称（只用repo部分，不用owner）
        repo_name = repo.full_name.split('/')[-1]
        
        # 1. 获取README
        readme_content, readme_filename = self.get_readme(repo.full_name)
        
        if not readme_content:
            return ProcessResult(
                repo_name=repo.full_name,
                success=False,
                readme_found=False,
                tasks_generated=False,
                num_tasks=0,
                error="README未找到"
            )
        
        # 注意：README 将在验证通过后保存
        
        # 2. 提取代码块（setup、docker、示例代码）
        code_blocks = self.extract_code_blocks(readme_content)
        logger.info(f"提取代码块: setup={len(code_blocks['setup_commands'])}, "
                   f"docker={len(code_blocks['docker_commands'])}, "
                   f"example={len(code_blocks['example_code'])}")
        
        # 检查是否找到示例代码，如果没有则跳过（setup和docker不算）
        if len(code_blocks['example_code']) == 0:
            logger.warning(f"未找到示例代码，跳过仓库: {repo.full_name}")
            return ProcessResult(
                repo_name=repo.full_name,
                success=False,
                readme_found=True,
                tasks_generated=False,
                num_tasks=0,
                error="README中未找到示例代码"
            )
        
        # 3. 【第一步验证】检查代码是否是repo的usage example
        logger.info("开始验证代码是否为usage example...")
        is_usage_example = self.validate_code_is_usage_example(
            readme_content, 
            code_blocks['example_code'], 
            repo.full_name
        )
        
        if not is_usage_example:
            logger.warning(f"代码不是usage example，跳过仓库: {repo.full_name}")
            return ProcessResult(
                repo_name=repo.full_name,
                success=False,
                readme_found=True,
                tasks_generated=False,
                num_tasks=0,
                error="代码块不是仓库的使用示例"
            )
        
        # 4. 获取Docker文件
        docker_files = self.get_docker_files(repo.full_name)
        if docker_files:
            logger.info(f"找到Docker文件: {list(docker_files.keys())}")
        
        # 5. 【第二步】生成任务（只使用example_code）
        tasks_data = self.generate_tasks_openai(code_blocks['example_code'], repo.full_name)
        
        if not tasks_data:
            return ProcessResult(
                repo_name=repo.full_name,
                success=False,
                readme_found=True,
                tasks_generated=False,
                num_tasks=0,
                error="任务生成失败"
            )
        
        # 提取任务列表
        tasks_list = tasks_data.get("tasks", [])
        if isinstance(tasks_data, list):
            tasks_list = tasks_data
        
        # 过滤掉所有关键字段都是null的任务
        filtered_tasks = []
        for task in tasks_list:
            # 检查是否所有关键字段都是null
            if not all([
                task.get('example_code') is None,
                task.get('running_command') is None,
                task.get('expected_input') is None,
                task.get('expected_output') is None
            ]):
                # 至少有一个字段不是null，保留这个任务
                filtered_tasks.append(task)
            else:
                logger.debug(f"过滤掉无效任务: {task.get('task_title', 'Unknown')}")
        
        tasks_list = filtered_tasks
        
        # 检查任务列表是否为空，如果为空则跳过
        if not tasks_list or len(tasks_list) == 0:
            logger.warning(f"过滤后任务列表为空，跳过仓库: {repo.full_name}")
            return ProcessResult(
                repo_name=repo.full_name,
                success=False,
                readme_found=True,
                tasks_generated=False,
                num_tasks=0,
                error="过滤后任务列表为空"
            )
        
        # 提取input_to_gpt（用于调试）
        input_to_gpt = tasks_data.get('input_to_gpt', {})
        
        # 6. 保存README（只有成功生成任务的才保存）
        readme_file = self.readmes_dir / f"{repo_name}_{readme_filename}"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # 7. 保存任务（包含setup和docker信息）
        complete_tasks_data = {
            "tasks": tasks_list,
            "setup": {
                "setup_commands": code_blocks['setup_commands'],
                "docker_commands": code_blocks['docker_commands'],
                "docker_files": docker_files
            },
            "input_to_gpt": input_to_gpt  # 添加调试信息
        }
        
        task_file = self.tasks_dir / f"{repo_name}_tasks.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(complete_tasks_data, f, indent=2, ensure_ascii=False)
        
        # 8. 保存Docker文件到独立目录
        if docker_files:
            docker_dir = self.output_dir / "docker_files" / repo_name
            docker_dir.mkdir(parents=True, exist_ok=True)
            
            for filename, content in docker_files.items():
                docker_file_path = docker_dir / filename
                with open(docker_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        # 9. 保存元数据
        metadata = {
            "repo": repo.to_dict(),
            "readme_file": str(readme_file),
            "task_file": str(task_file),
            "num_tasks": len(tasks_list),
            "num_setup_commands": len(code_blocks['setup_commands']),
            "num_docker_commands": len(code_blocks['docker_commands']),
            "has_docker_files": bool(docker_files),
            "docker_files_list": list(docker_files.keys()) if docker_files else [],
            "processed_at": datetime.now().isoformat()
        }
        
        metadata_file = self.metadata_dir / f"{repo_name}_meta.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 10. 追加到数据集文件（JSONL格式 - 以 repo name 为 key）
        dataset_entry = {
            "repo_name": repo.full_name,
            "stars": repo.stars,
            "language": repo.language,
            "tasks": tasks_list,
            "setup": {
                "setup_commands": code_blocks['setup_commands'],
                "docker_commands": code_blocks['docker_commands'],
                "has_docker_files": bool(docker_files)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.dataset_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dataset_entry, ensure_ascii=False) + '\n')
        
        logger.info(f"处理完成: {repo.full_name} - {len(tasks_list)} 个任务")
        
        return ProcessResult(
            repo_name=repo.full_name,
            success=True,
            readme_found=True,
            tasks_generated=True,
            num_tasks=len(tasks_list)
        )
    
    def run(self, 
            search_query: str = "stars:>1000 language:python",
            max_repos: int = 50,
            delay_seconds: float = 2.0):
        """
        运行批量处理
        
        Args:
            search_query: GitHub搜索查询
            max_repos: 最大处理仓库数（只计算有效处理的repo）
            delay_seconds: 请求间隔（秒）
        """
        logger.info("="*60)
        logger.info("批量处理开始")
        logger.info("="*60)
        
        # 验证API密钥
        if not self.openai_api_key:
            logger.error("缺少OpenAI API密钥！请设置OPENAI_API_KEY环境变量")
            return
        
        # 处理仓库，直到获得足够的有效repo
        results = []
        processed_repos = []
        valid_count = 0
        page = 1
        per_page = 30
        candidate_count = 0
        
        while valid_count < max_repos:
            # 搜索一批候选仓库
            logger.info(f"\n正在搜索第 {page} 页候选仓库...")
            
            params = {
                "q": search_query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page
            }
            
            try:
                response = requests.get(
                    f"{self.github_api}/search/repositories",
                    headers=self.github_headers,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                items = data.get("items", [])
                
                if not items:
                    logger.warning(f"\n没有更多候选仓库了。已找到 {valid_count} 个有效仓库，目标是 {max_repos} 个")
                    break
                
                logger.info(f"获取到 {len(items)} 个候选仓库")
                
                # 处理这批候选仓库
                for item in items:
                    if valid_count >= max_repos:
                        logger.info(f"\n✅ 已达到目标数量 {max_repos} 个有效仓库")
                        break
                    
                    candidate_count += 1
                    
                    repo = RepoMetadata(
                        full_name=item["full_name"],
                        stars=item["stargazers_count"],
                        forks=item["forks_count"],
                        language=item.get("language", "Unknown"),
                        description=item.get("description", ""),
                        topics=item.get("topics", []),
                        url=item["html_url"],
                        created_at=item["created_at"],
                        updated_at=item["updated_at"]
                    )
                    
                    logger.info(f"\n候选进度: {candidate_count} | 有效仓库: {valid_count}/{max_repos}")
                    logger.info(f"当前仓库: {repo.full_name} ({repo.stars} stars)")
                    
                    try:
                        result = self.process_repo(repo)
                        
                        # 只有成功处理的repo才计入
                        if result.success:
                            results.append(result)
                            processed_repos.append(repo)
                            valid_count += 1
                            
                            # 打印统计
                            logger.info(f"✓ 有效仓库累计: {valid_count}/{max_repos}")
                        else:
                            # 跳过的repo记录原因但不计入结果
                            logger.warning(f"✗ 跳过仓库 {repo.full_name}: {result.error}")
                        
                    except Exception as e:
                        logger.error(f"处理仓库异常 {repo.full_name}: {e}", exc_info=True)
                        # 异常的repo也不计入
                        logger.warning(f"✗ 跳过异常仓库: {repo.full_name}")
                    
                    # 请求间隔
                    time.sleep(delay_seconds)
                
                # 如果已经达到目标，退出外层循环
                if valid_count >= max_repos:
                    break
                
                # 翻页
                page += 1
                time.sleep(1)  # 翻页间隔
                
            except Exception as e:
                logger.error(f"搜索仓库失败: {e}")
                break
        
        # 生成汇总报告（只包含有效的repo）
        self.generate_summary(processed_repos, results)
        
        logger.info("="*60)
        logger.info("批量处理完成")
        logger.info(f"有效仓库: {valid_count}/{max_repos}")
        logger.info(f"总共尝试: {candidate_count} 个候选")
        logger.info("="*60)
    
    def generate_summary(self, repos: List[RepoMetadata], results: List[ProcessResult]):
        """
        生成汇总报告
        
        Args:
            repos: 仓库列表
            results: 结果列表
        """
        summary = {
            "total_repos": len(repos),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "readme_found": sum(1 for r in results if r.readme_found),
            "tasks_generated": sum(1 for r in results if r.tasks_generated),
            "total_tasks": sum(r.num_tasks for r in results),
            "average_tasks_per_repo": sum(r.num_tasks for r in results) / len(results) if results else 0,
            "language_distribution": {},
            "error_types": {},
            "top_repos": [],
            "generated_at": datetime.now().isoformat()
        }
        
        # 语言分布
        for repo in repos:
            lang = repo.language or "Unknown"
            summary["language_distribution"][lang] = summary["language_distribution"].get(lang, 0) + 1
        
        # 错误类型统计
        for result in results:
            if not result.success and result.error:
                summary["error_types"][result.error] = summary["error_types"].get(result.error, 0) + 1
        
        # Top仓库（按stars排序）
        sorted_repos = sorted(repos, key=lambda x: x.stars, reverse=True)[:10]
        summary["top_repos"] = [
            {"name": r.full_name, "stars": r.stars, "description": r.description}
            for r in sorted_repos
        ]
        
        # 保存汇总
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存详细结果
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        
        # 打印汇总
        print("\n" + "="*60)
        print("执行汇总")
        print("="*60)
        print(f"总仓库数: {summary['total_repos']}")
        print(f"成功处理: {summary['successful']}")
        print(f"处理失败: {summary['failed']}")
        print(f"成功率: {summary['successful']/summary['total_repos']*100:.1f}%")
        print(f"生成任务总数: {summary['total_tasks']}")
        print(f"平均每仓库任务数: {summary['average_tasks_per_repo']:.1f}")
        print(f"\n语言分布:")
        for lang, count in sorted(summary['language_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count}")
        print(f"\n汇总已保存: {summary_file}")
        print("="*60 + "\n")


def load_config(config_path: str = "config.yaml") -> Dict:
    """加载配置文件"""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"已加载配置文件: {config_path}")
            return config or {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def main():
    """主函数"""
    # 加载配置文件
    config = load_config()
    
    # 从配置文件获取设置
    search_config = config.get('search', {})
    processing_config = config.get('processing', {})
    output_config = config.get('output', {})
    openai_config = config.get('openai', {})
    presets = config.get('presets', {})
    
    # 从环境变量或配置获取API密钥
    github_token = os.getenv("GITHUB_TOKEN") or config.get('api', {}).get('github_token')
    openai_api_key = os.getenv("OPENAI_API_KEY") or config.get('api', {}).get('openai_api_key')
    
    # 确定搜索查询
    search_query = search_config.get('default_query', "stars:>1000 language:python")
    logger.info(f"使用搜索查询: {search_query}")
    
    # 获取代码块语言列表（从配置文件或使用默认值）
    code_languages = config.get('code_languages', ['python', 'sh', 'bash', 'console', 'shell', 'cmd', 'powershell'])
    
    # 创建处理器
    processor = BatchRepoProcessor(
        github_token=github_token,
        openai_api_key=openai_api_key,
        output_dir=output_config.get('base_dir', './dataset'),
        code_block_languages=code_languages
    )
    
    logger.info(f"输出目录: {output_config.get('base_dir', './dataset')}")
    logger.info(f"代码块语言过滤: {', '.join(code_languages)}")
    
    # 运行处理
    processor.run(
        search_query=search_query,
        max_repos=search_config.get('max_results', 50),
        delay_seconds=processing_config.get('delay_seconds', 2.0)
    )


if __name__ == "__main__":
    main()