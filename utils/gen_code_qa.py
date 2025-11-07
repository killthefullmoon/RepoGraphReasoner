#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate *unique-answer* English multi-hop (≥2 hops) QA pairs
from a code knowledge graph built with build_code_kg.py.
"""

import json
import pickle
import networkx as nx
from collections import defaultdict

INPUT_GRAPH = "graph.pkl"
OUTPUT_QA = "multi_hop_qas_en_unique.jsonl"


def load_graph(path=INPUT_GRAPH):
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
                "path": [m, lib, f, ext],
                "edges": [
                    [m, lib, "IMPORTS"],
                    [f, m, "IN_MODULE"],
                    [f, ext, "CALLS"]
                ]
            },
            "num_hops": 3,
            "verified": True
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
        qid += 1
        qas.append({
            "id": f"qa_{qid:05d}",
            "question": (
                f"Which test function calls a function that itself calls the external API `{ext_name}`?"
            ),
            "answer": [test],
            "answer_type": "Test",
            "graph_support": {
                "path": [test, fn, ext],
                "edges": [
                    [test, fn, "CALLS"],
                    [fn, ext, "CALLS"]
                ]
            },
            "num_hops": 2,
            "verified": True
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
                "path": [m, lib, f, callee],
                "edges": [
                    [m, lib, "IMPORTS"],
                    [f, m, "IN_MODULE"],
                    [f, callee, "CALLS"]
                ]
            },
            "num_hops": 3,
            "verified": True
        })

    return qas


if __name__ == "__main__":
    G = load_graph()
    qas = generate_unique_qas(G)
    with open(OUTPUT_QA, "w", encoding="utf-8") as f:
        for q in qas:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"[OK] Generated {len(qas)} unique multi-hop QA pairs → {OUTPUT_QA}")
