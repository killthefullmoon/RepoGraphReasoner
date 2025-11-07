import os
import argparse
import subprocess
from github import Github

# =============================
# 配置区
# =============================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # 建议使用个人token（避免API限流）
SAVE_DIR = "../dataset/repos/"
MIN_STARS = 100
SIZE_RANGE = "500..20000"  # KB
EXCLUDE_TOPICS = ["machine-learning", "deep-learning", "pytorch", "tensorflow", "ml"]

# =============================
# 工具函数
# =============================

def build_query():
    query = f"language:Python stars:>{MIN_STARS} size:{SIZE_RANGE}"
    for t in EXCLUDE_TOPICS:
        query += f" -topic:{t}"
    return query

def clone_repo(full_name, save_dir):
    repo_dir = os.path.join(save_dir, full_name.replace("/", "__"))
    if os.path.exists(repo_dir):
        print(f"⚠️ Already exists: {repo_dir}")
        return False
    url = f"https://github.com/{full_name}.git"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, repo_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"✅ Cloned {full_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to clone {full_name}")
        return False

# =============================
# 主逻辑
# =============================

def main(n_repos):
    os.makedirs(SAVE_DIR, exist_ok=True)
    gh = Github(GITHUB_TOKEN)
    query = build_query()

    print(f"🔍 Query: {query}")
    results = gh.search_repositories(query, sort="stars", order="desc")

    count = 0
    for repo in results:
        if count >= n_repos:
            break
        if repo.archived or repo.fork:
            continue
        if clone_repo(repo.full_name, SAVE_DIR):
            count += 1
    print(f"\n✅ Done. Downloaded {count} repos to {SAVE_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of repos to download")
    args = parser.parse_args()
    main(args.n)
