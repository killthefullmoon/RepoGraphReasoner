#!/usr/bin/env python3
"""
自动化脚本：读取README文件，提交给OpenAI，并获取任务描述
默认使用OpenAI API生成任务
"""

import os
import json
import requests
from pathlib import Path

class ReadmeTaskGenerator:
    def __init__(self, codebase_path=None):
        self.codebase_path = codebase_path or os.getcwd()
        
    def find_readme_file(self):
        """在codebase中查找README文件"""
        readme_candidates = [
            "README.rst",
            "README.md", 
            "README.txt",
            "README"
        ]
        
        for candidate in readme_candidates:
            readme_path = Path(self.codebase_path) / candidate
            if readme_path.exists():
                return readme_path
        
        return None

    def read_readme_content(self, readme_path):
        """读取README文件内容"""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"读取README文件时出错: {e}")
            return None

    def analyze_readme_examples(self, readme_content):
        """分析README中的示例代码"""
        examples = []
        
        # 查找代码块
        lines = readme_content.split('\n')
        in_code_block = False
        current_example = []
        
        for line in lines:
            if line.strip().startswith('>>>') or line.strip().startswith('...'):
                in_code_block = True
                current_example.append(line.strip())
            elif in_code_block and line.strip() == '':
                if current_example:
                    examples.append('\n'.join(current_example))
                    current_example = []
                in_code_block = False
            elif in_code_block:
                current_example.append(line.strip())
        
        return examples

    def submit_to_openai(self, readme_content, api_key):
        """将README内容提交给OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建提示词
        prompt = f"""
        请分析以下Python库的README文件内容，并描述其中的示例代码所执行的所有任务。有可能包含多个任务，请都描述出来。

        README内容：
        {readme_content}

        请基于README中的示例代码和功能描述，描述示例代码所执行的所有任务。任务应该：
        1. 利用该库的主要功能
        2. 包含实际的代码实现
        3. 有明确的目标和输出

        请以a list of JSON格式返回，每个JSON包含以下字段：
        - task_title: 任务标题
        - task_description: 任务详细描述
        - expected_output: 期望输出
        - expected_input: 期望输入

        在json之后，用自然语言描述所有任务。
        """
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return None
        except KeyError as e:
            print(f"API响应格式错误: {e}")
            return None

    def save_task_to_file(self, task_content, output_file="generated_task.txt"):
        """将生成的任务保存到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(task_content)
            
            print(f"任务已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def run(self, api_key=None):
        """运行主流程"""
        print("=== README自动化任务生成器 ===\n")
        
        print(f"正在搜索codebase: {self.codebase_path}")
        
        # 查找README文件
        readme_path = self.find_readme_file()
        if not readme_path:
            print("未找到README文件！")
            return
        
        print(f"找到README文件: {readme_path}")
        
        # 读取README内容
        readme_content = self.read_readme_content(readme_path)
        if not readme_content:
            print("无法读取README文件内容！")
            return
        
        print(f"README文件大小: {len(readme_content)} 字符")
        
        # 分析示例代码
        examples = self.analyze_readme_examples(readme_content)
        print(f"找到 {len(examples)} 个代码示例")
        
        # 获取API密钥
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("请设置环境变量 OPENAI_API_KEY")
                api_key = input("请输入你的OpenAI API密钥: ").strip()
                if not api_key:
                    print("未提供API密钥，退出程序")
                    return
        
        # 提交给OpenAI
        print("正在提交给OpenAI...")
        task_result = self.submit_to_openai(readme_content, api_key)
        if not task_result:
            print("从OpenAI获取任务失败！")
            return
        
        print("成功生成任务描述！")
        print("\n=== 生成的任务 ===")
        print(task_result)
        
        # 保存到文件
        self.save_task_to_file(task_result)
        
        print("\n=== 完成 ===")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='README自动化任务生成器')
    parser.add_argument('path', nargs='?', help='项目路径（可选，默认为当前目录）', default=None)
    parser.add_argument('--api-key', '-k', help='OpenAI API密钥')
    
    args = parser.parse_args()
    
    # 如果没有提供路径，使用当前目录
    codebase_path = args.path or os.getcwd()
    
    generator = ReadmeTaskGenerator(codebase_path)
    generator.run(api_key=args.api_key)

if __name__ == "__main__":
    main()
