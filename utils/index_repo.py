# tools/index_repo.py
import os
import ast
import json
import argparse
from pathlib import Path

def safe_read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def get_module_name(repo_root, file_path):
    rel = os.path.relpath(file_path, repo_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return rel_no_ext.replace(os.sep, ".")  # flask/app.py -> flask.app

def extract_imports(tree):
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports

def extract_calls(tree):
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return calls

def get_parent_name(tree, node):
    for p in ast.walk(tree):
        body = getattr(p, "body", None)
        if isinstance(body, list) and node in body:
            return getattr(p, "name", None)
    return None

def detect_test_flag(file_path, module_name, tree):
    path_lower = file_path.lower()
    if "test" in path_lower:
        return True

    imports = extract_imports(tree)
    if any(i.startswith(("pytest", "unittest", "nose", "hypothesis")) for i in imports):
        return True

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name.lower().startswith("test"):
                return True
    return False

def extract_symbols(file_path, repo_root):
    code = safe_read(file_path)
    if not code:
        return [], None
    try:
        tree = ast.parse(code)
    except Exception:
        return [], None

    module_name = get_module_name(repo_root, file_path)
    imports = extract_imports(tree)
    is_test = detect_test_flag(file_path, module_name, tree)

    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parent = get_parent_name(tree, node)
            name = node.name
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
            doc = ast.get_docstring(node)
            decorators = [getattr(d, "id", getattr(d, "attr", None)) for d in node.decorator_list]
            calls = extract_calls(node)

            symbols.append({
                "symbol_id": f"{module_name}::{name}",
                "path": os.path.relpath(file_path, repo_root),
                "module": module_name,
                "name": name,
                "kind": kind,
                "parent": parent,
                "start_line": start,
                "end_line": end,
                "docstring": doc,
                "calls": calls[:30],
                "decorators": decorators,
                "num_calls": len(calls),
            })

    file_meta = {
        "module": module_name,
        "imports": imports,
        "num_defs": len(symbols),
        "num_imports": len(imports),
        "has_test_flag": is_test
    }

    return symbols, file_meta

def build_index(repo_root, output_dir):
    symbols_out = []
    imports_out = []
    for root, _, files in os.walk(repo_root):
        for f in files:
            if not f.endswith(".py"):
                continue
            file_path = os.path.join(root, f)
            symbols, file_meta = extract_symbols(file_path, repo_root)
            if file_meta:
                imports_out.append({
                    "path": os.path.relpath(file_path, repo_root),
                    **file_meta
                })
            symbols_out.extend(symbols)

    os.makedirs(output_dir, exist_ok=True)
    symbols_path = os.path.join(output_dir, "symbols.jsonl")
    imports_path = os.path.join(output_dir, "imports.jsonl")

    with open(symbols_path, "w", encoding="utf-8") as f:
        for s in symbols_out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(imports_path, "w", encoding="utf-8") as f:
        for i in imports_out:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

    print(f"✅ Indexed {len(symbols_out)} symbols across {len(imports_out)} files.")
    print(f"   symbols → {symbols_path}")
    print(f"   imports → {imports_path}")
    return symbols_out, imports_out

if __name__ == "__main__":
    repo_base = Path(__file__).resolve().parents[1]
    default_repo = repo_base / "dataset" / "repos" / "flask"
    default_output = repo_base / "processed_data" / "flask" / "index"

    parser = argparse.ArgumentParser(description="Index a repository into JSONL artifacts.")
    parser.add_argument(
        "--repo-root",
        default=str(default_repo),
        help="Path to the repository to index (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        help="Directory to store imports.jsonl and symbols.jsonl (default: %(default)s)",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_dir = os.path.abspath(args.output_dir)
    build_index(repo_root, output_dir)
