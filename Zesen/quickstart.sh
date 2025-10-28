#!/bin/bash
# 快速启动脚本

echo "==================================="
echo "GitHub数据集Pipeline - 快速启动"
echo "==================================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

echo "✓ Python3已安装"

# 检查依赖
echo "检查依赖..."
python3 -c "import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖: requests"
    pip3 install requests
fi

echo "✓ 依赖已就绪"

# 检查API密钥
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠️  未设置 OPENAI_API_KEY 环境变量"
    echo "请运行: export OPENAI_API_KEY='your-key'"
    echo ""
    read -p "请输入OpenAI API密钥（或按Enter跳过）: " api_key
    if [ -n "$api_key" ]; then
        export OPENAI_API_KEY="$api_key"
        echo "✓ API密钥已设置"
    else
        echo "未设置API密钥，脚本可能无法运行"
        exit 1
    fi
else
    echo "✓ OpenAI API密钥已设置"
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ℹ️  未设置 GITHUB_TOKEN（可选，但推荐）"
else
    echo "✓ GitHub Token已设置"
fi

echo ""
echo "==================================="
echo "选择运行模式:"
echo "==================================="
echo "1. 测试模式（5个仓库）"
echo "2. 小规模（20个仓库）"
echo "3. 中等规模（50个仓库）"
echo "4. 大规模（100个仓库）"
echo "5. 自定义"
echo ""
read -p "请选择 [1-5]: " choice

case $choice in
    1)
        echo "运行测试模式..."
        python3 batch_process_repos.py --max-repos 5 --delay 2.0
        ;;
    2)
        echo "运行小规模处理..."
        python3 batch_process_repos.py --max-repos 20 --delay 2.0
        ;;
    3)
        echo "运行中等规模处理..."
        python3 batch_process_repos.py --max-repos 50 --delay 2.5
        ;;
    4)
        echo "运行大规模处理..."
        python3 batch_process_repos.py --max-repos 100 --delay 3.0
        ;;
    5)
        read -p "仓库数量: " num_repos
        read -p "延迟秒数: " delay
        python3 batch_process_repos.py --max-repos $num_repos --delay $delay
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "==================================="
echo "处理完成！"
echo "==================================="
echo "查看结果:"
echo "  汇总报告: dataset/summary.json"
echo "  详细日志: batch_process.log"
echo "  数据集: dataset/dataset.jsonl"
echo ""


