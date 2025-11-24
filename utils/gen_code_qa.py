#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate *unique-answer* English multi-hop (≥2 hops) QA pairs
from a code knowledge graph built with build_code_kg.py.
"""

import json
import pickle
import argparse
from pathlib import Path
import networkx as nx
from collections import defaultdict

_REPO_BASE = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path("/scratch/zmao_root/zmao98/boyuann/dataset/processed_data")

def build_node_file_map(G):
    """
    Map graph nodes to the file paths that define or contain them.
    """
    node_to_files = defaultdict(set)

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "File":
            continue
        path = data.get("path")
        if not path:
            continue

        module_name = data.get("module")
        if module_name:
            node_to_files[f"module:{module_name}"].add(path)

        for succ in G.successors(node):
            if "DEFINES" in G.edges[node, succ]["etype"]:
                node_to_files[succ].add(path)

    for node in G.nodes:
        for succ in G.successors(node):
            if (
                "IN_FILE" in G.edges[node, succ]["etype"]
                and G.nodes[succ].get("node_type") == "File"
            ):
                path = G.nodes[succ].get("path")
                if path:
                    node_to_files[node].add(path)

    return node_to_files


def gather_relevant_files(node_to_files, nodes):
    files = set()
    for node in nodes:
        files.update(node_to_files.get(node, ()))
    return sorted(files)


def longest_common_prefix_len(a_parts, b_parts):
    count = 0
    for a, b in zip(a_parts, b_parts):
        if a != b:
            break
        count += 1
    return count


def build_module_indexes(G, modules):
    module_names = {m: get_name(G, m) for m in modules}
    suffix_map = defaultdict(set)

    for m in modules:
        name = module_names[m]
        parts = name.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            suffix_map[suffix].add(m)

        if name.endswith(".__init__"):
            trimmed = name[: -len(".__init__")]
            trimmed_parts = trimmed.split(".")
            for i in range(len(trimmed_parts)):
                suffix = ".".join(trimmed_parts[i:])
                suffix_map[suffix].add(m)

    return module_names, suffix_map


def resolve_library_files(G, node_to_files, module_names, modules_by_suffix,
                          importer_module, library_node):
    lib_name = get_name(G, library_node)
    importer_name = module_names.get(importer_module)
    if not importer_name:
        return []

    candidates = set(modules_by_suffix.get(lib_name, set()))

    if not candidates and "." not in lib_name and "." in importer_name:
        parent = importer_name.rsplit(".", 1)[0]
        rel_suffix = f"{parent}.{lib_name}"
        candidates = set(modules_by_suffix.get(rel_suffix, set()))

    if not candidates:
        return []

    importer_parts = importer_name.split(".")
    best_score = -1
    best_ids = []
    for mid in candidates:
        cand_name = module_names[mid]
        score = longest_common_prefix_len(importer_parts, cand_name.split("."))
        if score > best_score:
            best_score = score
            best_ids = [mid]
        elif score == best_score:
            best_ids.append(mid)

    if best_score <= 0:
        return []

    expanded_ids = set(best_ids)
    for mid in best_ids:
        base_name = module_names[mid]
        if base_name.endswith(".__init__"):
            base_prefix = base_name[: -len(".__init__")]
            prefix = f"{base_prefix}."
            for cand_id, cand_name in module_names.items():
                if cand_name.startswith(prefix):
                    expanded_ids.add(cand_id)

    files = set()
    for mid in expanded_ids:
        files.update(node_to_files.get(mid, ()))
    return sorted(files)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_name(G, n):
    return G.nodes[n].get("name") or n


def ntype(G, n):
    return G.nodes[n].get("node_type")


def is_func(G, n): return ntype(G, n) in {"Function", "Test"}
def is_mod(G, n): return ntype(G, n) == "Module"
def is_lib(G, n): return ntype(G, n) == "Library"
def is_ext(G, n): return ntype(G, n) == "ExternalAPI"


def generate_unique_qas(G):
    qas, qid = [], 0

    modules = [n for n in G.nodes if is_mod(G, n)]
    functions = [n for n in G.nodes if is_func(G, n)]
    node_to_files = build_node_file_map(G)
    module_names, modules_by_suffix = build_module_indexes(G, modules)

    # -------------------------------
    # Pattern 1: Module IMPORTS Library -> Function IN_MODULE -> Function CALLS ExternalAPI
    # -------------------------------
    cond_map = defaultdict(list)  # key = (lib_name, ext_name) -> [function_ids]

    for m in modules:
        libs = [l for l in G.successors(m) if "IMPORTS" in G.edges[m, l]["etype"]]
        funcs = [f for f in G.predecessors(m) if "IN_MODULE" in G.edges[f, m]["etype"]]
        for f in funcs:
            called_exts = [t for t in G.successors(f)
                           if "CALLS" in G.edges[f, t]["etype"] and is_ext(G, t)]
            for lib in libs:
                for ext in called_exts:
                    key = (get_name(G, lib), get_name(G, ext))
                    cond_map[key].append((m, lib, f, ext))

    for (lib_name, ext_name), paths in cond_map.items():
        if len(paths) != 1:
            continue  # only unique answers allowed
        m, lib, f, ext = paths[0]
        path_nodes = [m, lib, f, ext]
        lib_files = resolve_library_files(
            G, node_to_files, module_names, modules_by_suffix, m, lib
        )
        relevant_files = set(gather_relevant_files(node_to_files, path_nodes))
        relevant_files.update(lib_files)
        qid += 1
        qas.append({
            "id": f"qa_{qid:05d}",
            "question": (
                f"Which function is defined in a module that imports the library `{lib_name}`, "
                f"and calls the external API `{ext_name}`?"
            ),
            "answer": [f],
            "answer_type": ntype(G, f),
            "graph_support": {
                "path": path_nodes,
                "edges": [
                    [m, lib, "IMPORTS"],
                    [f, m, "IN_MODULE"],
                    [f, ext, "CALLS"]
                ]
            },
            "num_hops": 3,
            "verified": True,
            "relavant_files": sorted(relevant_files)
        })

    # -------------------------------
    # Pattern 2: Test CALLS Function -> Function CALLS ExternalAPI
    # -------------------------------
    cond_map = defaultdict(list)  # key = (ext_name) -> [test_nodes]

    for test in functions:
        if ntype(G, test) != "Test":
            continue
        for fn in G.successors(test):
            if "CALLS" not in G.edges[test, fn]["etype"] or not is_func(G, fn):
                continue
            externs = [t for t in G.successors(fn)
                       if "CALLS" in G.edges[fn, t]["etype"] and is_ext(G, t)]
            for ext in externs:
                key = get_name(G, ext)
                cond_map[key].append((test, fn, ext))

    for ext_name, paths in cond_map.items():
        if len(paths) != 1:
            continue
        test, fn, ext = paths[0]
        path_nodes = [test, fn, ext]
        relevant_files = gather_relevant_files(node_to_files, path_nodes)
        qid += 1
        qas.append({
            "id": f"qa_{qid:05d}",
            "question": (
                f"Which test function calls a function that itself calls the external API `{ext_name}`?"
            ),
            "answer": [test],
            "answer_type": "Test",
            "graph_support": {
                "path": path_nodes,
                "edges": [
                    [test, fn, "CALLS"],
                    [fn, ext, "CALLS"]
                ]
            },
            "num_hops": 2,
            "verified": True,
            "relavant_files": relevant_files
        })

    # -------------------------------
    # Pattern 3: Module IMPORTS Library -> Function IN_MODULE -> Function CALLS another Function
    # -------------------------------
    cond_map = defaultdict(list)  # key = (lib_name, callee_name) -> [fn_nodes]

    for m in modules:
        libs = [l for l in G.successors(m) if "IMPORTS" in G.edges[m, l]["etype"]]
        funcs = [f for f in G.predecessors(m) if "IN_MODULE" in G.edges[f, m]["etype"]]
        for f in funcs:
            callees = [t for t in G.successors(f)
                       if "CALLS" in G.edges[f, t]["etype"] and is_func(G, t)]
            for lib in libs:
                for callee in callees:
                    key = (get_name(G, lib), get_name(G, callee))
                    cond_map[key].append((m, lib, f, callee))

    for (lib_name, callee_name), paths in cond_map.items():
        if len(paths) != 1:
            continue
        m, lib, f, callee = paths[0]
        path_nodes = [m, lib, f, callee]
        lib_files = resolve_library_files(
            G, node_to_files, module_names, modules_by_suffix, m, lib
        )
        relevant_files = set(gather_relevant_files(node_to_files, path_nodes))
        relevant_files.update(lib_files)
        qid += 1
        qas.append({
            "id": f"qa_{qid:05d}",
            "question": (
                f"Which function is defined in a module that imports the library `{lib_name}`, "
                f"and calls another function named `{callee_name}`?"
            ),
            "answer": [f],
            "answer_type": ntype(G, f),
            "graph_support": {
                "path": path_nodes,
                "edges": [
                    [m, lib, "IMPORTS"],
                    [f, m, "IN_MODULE"],
                    [f, callee, "CALLS"]
                ]
            },
            "num_hops": 3,
            "verified": True,
            "relavant_files": sorted(relevant_files)
        })

    return qas


def write_qas(graph_path, output_path):
    G = load_graph(graph_path)
    qas = generate_unique_qas(G)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for q in qas:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return len(qas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-hop QA pairs from a code KG.")
    parser.add_argument("--graph", help="Path to a specific graph.pkl file")
    parser.add_argument("--output", help="Destination JSONL file when --graph is used")
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Root directory containing processed repositories with graph.pkl files",
    )
    parser.add_argument(
        "--graph-name",
        default="graph.pkl",
        help="Graph filename to look for inside each processed repository",
    )
    parser.add_argument(
        "--qa-name",
        default="multi_hop_qas_en_unique.jsonl",
        help="QA filename to write inside each processed repository",
    )
    args = parser.parse_args()

    if args.graph:
        output_path = Path(args.output) if args.output else Path(args.graph).with_name(args.qa_name)
        count = write_qas(Path(args.graph), output_path)
        print(f"[OK] Generated {count} unique multi-hop QA pairs → {output_path}")
    else:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        processed = 0
        for repo_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            graph_path = repo_dir / args.graph_name
            if not graph_path.is_file():
                continue
            output_path = repo_dir / args.qa_name
            count = write_qas(graph_path, output_path)
            processed += 1
            print(f"[OK] {repo_dir.name}: {count} QA pairs")

        if processed == 0:
            print(f"[WARN] No repositories with {args.graph_name} found in {dataset_dir}")
        else:
            print(f"Processed {processed} repositories under {dataset_dir}")
