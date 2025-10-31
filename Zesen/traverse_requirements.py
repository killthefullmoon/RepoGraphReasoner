#!/usr/bin/env python3
"""
Recursively explore a repo's Python dependencies by following requirements files,
and optionally render a sequential dependency graph.

Usage:
    python traverse_requirements.py /path/to/repo \
        [--max-depth 2] \
        [--cache-dir .deps_cache] \
        [--output deps.json] \
        [--visualize] [--graph-output graph.html]

Constraints:
    - Only follows dependencies that expose a publicly accessible GitHub repository.
    - Stops at the specified recursion depth (default: 1).
    - Parses common requirements files: requirements*.txt, setup.cfg, pyproject.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

# ---------------------------------------------------------------------------
# Data classes


@dataclass
class Requirement:
    name: str
    specifier: Optional[str] = None

    @classmethod
    def parse(cls, line: str) -> Optional["Requirement"]:
        line = line.strip()
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line or line.startswith("#"):
            return None
        if line.startswith("-e ") or line.startswith("--"):
            return None
        parts = re.split(r"\s*([!<>=~]{1,2})\s*", line, maxsplit=1)
        if len(parts) == 3:
            name, op, version = parts
            return cls(name=name.strip(), specifier=f"{op}{version.strip()}")
        return cls(name=line)


@dataclass
class RepoNode:
    repo_name: str
    repo_url: str
    requirements_files: List[str]
    depth: int
    dependencies: List["RepoNode"]
    missing_requirements: bool = False
    note: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "requirements_files": self.requirements_files,
            "depth": self.depth,
            "missing_requirements": self.missing_requirements,
            "note": self.note,
            "dependencies": [child.to_dict() for child in self.dependencies],
        }


# ---------------------------------------------------------------------------
# Helpers


GITHUB_RE = re.compile(r"https://github\.com/([\w\-]+)/([\w\-.]+)")


def run(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    if verbose:
        location = f" (cwd={cwd})" if cwd else ""
        print(f"→ Running: {' '.join(cmd)}{location}")
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def find_requirement_files(repo_root: Path) -> List[Path]:
    candidates = []
    for pattern in [
        "requirements.txt",
        "requirements-*.txt",
        "requirements/*.txt",
        "setup.cfg",
        "pyproject.toml",
    ]:
        candidates.extend(repo_root.glob(pattern))
    return [p for p in candidates if p.is_file()]


def parse_requirements_file(path: Path) -> List[Requirement]:
    reqs: List[Requirement] = []
    if path.name == "setup.cfg":
        try:
            import configparser

            cfg = configparser.ConfigParser()
            cfg.read(path)
            if cfg.has_option("options", "install_requires"):
                for line in cfg.get("options", "install_requires").splitlines():
                    req = Requirement.parse(line)
                    if req:
                        reqs.append(req)
        except Exception:
            pass
        return reqs

    if path.name == "pyproject.toml":
        try:
            import tomllib

            data = tomllib.loads(path.read_text())
            deps = data.get("project", {}).get("dependencies", [])
            for dep in deps:
                req = Requirement.parse(dep)
                if req:
                    reqs.append(req)
        except Exception:
            pass
        return reqs

    try:
        for line in path.read_text().splitlines():
            req = Requirement.parse(line)
            if req:
                reqs.append(req)
    except UnicodeDecodeError:
        pass
    return reqs


def fetch_pypi_metadata(package: str) -> Optional[Dict]:
    url = f"https://pypi.org/pypi/{package}/json"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except requests.exceptions.JSONDecodeError:
        return None


def locate_github_repo(metadata: Dict) -> Optional[str]:
    info = metadata.get("info", {}) if metadata else {}
    urls = info.get("project_urls", {}) or {}
    candidates = [
        info.get("project_url"),
        info.get("home_page"),
        info.get("package_url"),
        info.get("project_url"),
    ]
    candidates.extend(urls.values())
    for url in candidates:
        if not url:
            continue
        match = GITHUB_RE.search(url)
        if match:
            owner, repo = match.groups()
            if owner.lower() == "sponsors":
                continue
            return f"https://github.com/{owner}/{repo}"
    return None


def clone_repo(repo_url: str, dest_dir: Path, verbose: bool = False) -> Optional[Path]:
    repo_name = repo_url.rstrip("/").split("/")[-1]
    target = dest_dir / repo_name
    if target.exists():
        return target
    if verbose:
        print(f"↳ Cloning {repo_url} into {target}")
    result = run(["git", "clone", "--depth", "1", repo_url, str(target)], verbose=verbose)
    if result.returncode != 0:
        if verbose:
            print(f"⚠️  git clone failed for {repo_url}: {result.stderr.strip()}")
        return None
    if verbose:
        print(f"✅ Cloned {repo_url}")
    return target


# ---------------------------------------------------------------------------
# Recursive traversal


def traverse_repo(
    repo_root: Path,
    repo_url: str,
    depth: int,
    max_depth: int,
    cache_dir: Path,
    visited: Set[str],
    verbose: bool = False,
) -> RepoNode:
    if verbose:
        print(f"\n[depth {depth}] Processing repo: {repo_root} ({repo_url or 'local'})")
    requirements_paths = find_requirement_files(repo_root)
    missing_requirements = len(requirements_paths) == 0
    dependencies: List[RepoNode] = []

    if verbose:
        if missing_requirements:
            print("  • No requirements files found.")
        else:
            print("  • Found requirements files:")
            for path in requirements_paths:
                print(f"    - {path.relative_to(repo_root)}")

    if depth < max_depth and not missing_requirements:
        packages: List[Requirement] = []
        for path in requirements_paths:
            packages.extend(parse_requirements_file(path))

        for requirement in packages:
            package_name = requirement.name
            if package_name.lower() in visited:
                if verbose:
                    print(f"  • Skipping already visited package: {package_name}")
                continue
            if verbose:
                print(f"  • Resolving package: {package_name}")
            metadata = fetch_pypi_metadata(package_name)
            if not metadata:
                if verbose:
                    print(f"    ↳ PyPI metadata unavailable for {package_name}")
                continue
            repo_link = locate_github_repo(metadata)
            if not repo_link:
                if verbose:
                    print(f"    ↳ No public GitHub repository for {package_name}")
                continue

            visited.add(package_name.lower())
            cloned = clone_repo(repo_link, cache_dir, verbose=verbose)
            if not cloned:
                if verbose:
                    print(f"    ↳ Failed to clone {repo_link}, skipping.")
                continue
            if verbose:
                print(f"    ↳ Descending into {repo_link}")
            child_node = traverse_repo(
                repo_root=cloned,
                repo_url=repo_link,
                depth=depth + 1,
                max_depth=max_depth,
                cache_dir=cache_dir,
                visited=visited,
                verbose=verbose,
            )
            dependencies.append(child_node)

    repo_name = repo_url.rstrip("/").split("/")[-1] if repo_url else repo_root.name
    return RepoNode(
        repo_name=repo_name,
        repo_url=repo_url,
        requirements_files=[str(p.relative_to(repo_root)) for p in requirements_paths],
        depth=depth,
        dependencies=dependencies,
        missing_requirements=missing_requirements,
    )


# ---------------------------------------------------------------------------
# Visualization helpers


def flatten_dependency_tree(root: RepoNode) -> Tuple[List[RepoNode], List[Tuple[int, int]]]:
    """Linearize dependency tree into nodes list and parent-child edges."""
    nodes: List[RepoNode] = []
    edges: List[Tuple[int, int]] = []
    stack: List[Tuple[RepoNode, Optional[int]]] = [(root, None)]

    while stack:
        node, parent_idx = stack.pop()
        idx = len(nodes)
        nodes.append(node)
        if parent_idx is not None:
            edges.append((parent_idx, idx))
        for child in reversed(node.dependencies):
            stack.append((child, idx))

    return nodes, edges


def compute_sequential_positions(nodes: List[RepoNode]) -> Dict[int, Tuple[float, float]]:
    """Assign deterministic positions for sequential left-to-right layout."""
    by_depth: Dict[int, List[int]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        by_depth[node.depth].append(idx)

    positions: Dict[int, Tuple[float, float]] = {}
    x_step = 280.0
    y_step = 130.0

    for depth in sorted(by_depth):
        indices = sorted(by_depth[depth], key=lambda i: nodes[i].repo_name.lower())
        for order, node_idx in enumerate(indices):
            offset = order - (len(indices) - 1) / 2
            x = depth * x_step
            y = offset * y_step
            positions[node_idx] = (x, y)

    return positions


def render_dependency_graph_html(
    root: RepoNode,
    output_path: Path,
) -> None:
    """Render dependency tree as PyVis HTML graph."""
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise SystemExit("Missing dependency 'pyvis'. Install it via 'pip install pyvis'.") from exc

    nodes, edges = flatten_dependency_tree(root)
    positions = compute_sequential_positions(nodes)

    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.set_options(
        json.dumps(
            {
                "physics": {"enabled": False},
                "interaction": {
                    "dragNodes": True,
                    "navigationButtons": True,
                    "multiselect": True,
                },
            }
        )
    )

    for idx, node in enumerate(nodes):
        x, y = positions.get(idx, (0.0, 0.0))
        title_lines = [
            f"<b>{node.repo_name}</b>",
            f"URL: {node.repo_url or 'N/A'}",
            f"Depth: {node.depth}",
        ]
        if node.requirements_files:
            title_lines.append("Requirements:")
            title_lines.extend(node.requirements_files)
        if node.missing_requirements:
            title_lines.append("No requirements files found.")
        if node.note:
            title_lines.append(f"Note: {node.note}")

        net.add_node(
            idx,
            label=node.repo_name,
            title="<br>".join(title_lines),
            x=x,
            y=y,
            fixed=True,
            shape="box" if idx == 0 else "ellipse",
            color="#5B8FF9" if idx == 0 else "#5AD8A6",
        )

    for parent_idx, child_idx in edges:
        net.add_edge(parent_idx, child_idx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)
    print(f"✅ Dependency graph written to {output_path}")


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively traverse requirements to discover dependency repos.")
    parser.add_argument("repo", type=Path, help="Path to the root repository.")
    parser.add_argument("--repo-url", type=str, default="", help="Optional canonical URL for the root repo (for metadata).")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Recursion depth for dependency traversal (0 = only root).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".deps_cache"),
        help="Directory to store cloned dependency repos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the resulting dependency graph as JSON.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render an interactive dependency graph (saved to --graph-output or default path).",
    )
    parser.add_argument(
        "--graph-output",
        type=Path,
        default=None,
        help="HTML path for visualization (default: <repo>/_deps/dependency_graph.html).",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove the cache directory before traversal.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo.resolve()
    if not repo_root.exists():
        raise SystemExit(f"Repository path not found: {repo_root}")

    cache_dir = args.cache_dir.resolve()
    if args.clean_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    visited: Set[str] = set()
    root_node = traverse_repo(
        repo_root=repo_root,
        repo_url=args.repo_url or "",
        depth=0,
        max_depth=max(0, args.max_depth),
        cache_dir=cache_dir,
        visited=visited,
        verbose=args.verbose,
    )

    output = root_node.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        print(f"Dependency graph written to {args.output}")
    else:
        print(json.dumps(output, indent=2))

    if args.visualize or args.graph_output:
        graph_path = args.graph_output or (repo_root / "_deps" / "dependency_graph.html")
        render_dependency_graph_html(root_node, graph_path)


if __name__ == "__main__":
    main(sys.argv[1:])
