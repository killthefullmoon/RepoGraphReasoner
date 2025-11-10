#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a Code Knowledge Graph (KG) from imports.jsonl and symbols.jsonl.

Inputs:
  - imports.jsonl: per-file/per-module imports and file-level metadata
  - symbols.jsonl: per-symbol (function/class) info and call graph hints

Outputs:
  - graph.pkl:       networkx.DiGraph pickle
  - graph.graphml:   GraphML for Gephi or other tools
  - graph.json:      JSON with nodes/edges (for quick inspection)

Node types:
  - Module(name=module_str)
  - Library(name=import_str)
  - File(path=path_str)
  - Function(symbol_id or module::name)
  - Test(same id as function but node_type='Test')
  - ExternalAPI(name=callee_str that can't be resolved to a Function)

Edge types:
  - IMPORTS:    Module -> Library
  - DEFINES:    File -> {Function|Test}
  - IN_MODULE:  {Function|Test} -> Module
  - IN_FILE:    {Function|Test} -> File
  - CALLS:      {Function|Test} -> {Function|ExternalAPI}
  - DECORATED_BY: {Function|Test} -> ExternalAPI (e.g., "route")

Heuristics for tests:
  - file path contains "/tests/" OR module contains ".tests."
  - function name startswith "test_"
  - imports.jsonl line for that file has has_test_flag=True

Usage:
  python build_code_kg.py \
      --imports imports.jsonl \
      --symbols symbols.jsonl \
      --out-prefix graph

"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import networkx as nx


# ----------------------------
# Helpers
# ----------------------------

def add_node(G: nx.DiGraph, node_id: str, **attrs) -> None:
    """Add node with attributes; merge if exists."""
    if node_id in G:
        G.nodes[node_id].update(attrs)
    else:
        G.add_node(node_id, **attrs)


def add_edge(G: nx.DiGraph, s: str, t: str, **attrs) -> None:
    """Add edge and coalesce parallel edges by accumulating a set of 'etype'."""
    if G.has_edge(s, t):
        # merge edge attributes (keep unique etypes)
        for k, v in attrs.items():
            if k == "etype":
                prev = G.edges[s, t].get("etype")
                if isinstance(prev, set):
                    if isinstance(v, set):
                        prev |= v
                    else:
                        prev.add(v)
                    G.edges[s, t]["etype"] = prev
                else:
                    newset = set()
                    if prev: newset.add(prev)
                    if isinstance(v, set):
                        newset |= v
                    else:
                        newset.add(v)
                    G.edges[s, t]["etype"] = newset
            else:
                # last-write-wins for other attributes
                G.edges[s, t][k] = v
    else:
        # normalize etype to a set for easier merging later
        if "etype" in attrs and not isinstance(attrs["etype"], set):
            attrs["etype"] = {attrs["etype"]}
        G.add_edge(s, t, **attrs)


def is_test_context(path: str, module: str, has_test_flag: bool, fn_name: str) -> bool:
    return (
        "tests" + os.sep in path or "/tests/" in path or "\\tests\\" in path
        or ".tests." in module
        or fn_name.startswith("test_")
        or has_test_flag
    )


def function_node_id(symbol_id: str, module: str, name: str) -> str:
    """Use symbol_id if present; otherwise fallback to module::name."""
    if symbol_id:
        return symbol_id
    return f"{module}::{name}"


# ----------------------------
# Main build routine
# ----------------------------

def build_graph(imports_path: str, symbols_path: str) -> nx.DiGraph:
    G = nx.DiGraph()
    G.graph["kg"] = "code"
    G.graph["version"] = "0.1.0"

    # ---- Pass 1: read imports.jsonl -> create Module, File, Library, and IMPORTS edges
    file_test_flag: Dict[str, bool] = {}
    module_for_file: Dict[str, str] = {}

    with open(imports_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            path = rec.get("path", "")
            module = rec.get("module", "")
            imports = rec.get("imports", []) or []
            num_defs = rec.get("num_defs")
            num_imports = rec.get("num_imports")
            has_test_flag = bool(rec.get("has_test_flag", False))

            file_test_flag[path] = has_test_flag
            module_for_file[path] = module

            # Nodes
            add_node(G, f"module:{module}", node_type="Module", name=module)
            add_node(G, f"file:{path}", node_type="File", path=path,
                     module=module, num_defs=num_defs, num_imports=num_imports,
                     has_test_flag=has_test_flag)

            # IMPORTS: Module -> Library
            for lib in imports:
                add_node(G, f"lib:{lib}", node_type="Library", name=lib)
                add_edge(G, f"module:{module}", f"lib:{lib}", etype="IMPORTS")

    # ---- Pass 2: read symbols.jsonl -> create Function/Test and edges
    # We'll index function names inside a module to help resolve CALLS.
    fn_index: Dict[Tuple[str, str], str] = {}  # (module, name) -> node_id
    all_fn_names_in_module: Dict[str, set] = {}

    # First pass over symbols to create nodes and basic edges.
    symbols: list[dict] = []
    with open(symbols_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sym = json.loads(line)
            symbols.append(sym)

    # Create function/test nodes first
    for sym in symbols:
        kind = sym.get("kind")
        if kind not in ("Function", "Class"):  # focus on functions (and optionally classes)
            continue

        symbol_id = sym.get("symbol_id")  # examples.javascript.js_example.views::add
        path = sym.get("path", "")
        module = sym.get("module", "")
        name = sym.get("name", "")
        start_line = sym.get("start_line")
        end_line = sym.get("end_line")
        docstring = sym.get("docstring")
        decorators = sym.get("decorators") or []
        num_calls = sym.get("num_calls")

        test_ctx = is_test_context(
            path=path,
            module=module,
            has_test_flag=file_test_flag.get(path, False),
            fn_name=name
        )

        node_id = function_node_id(symbol_id, module, name)

        node_type = "Test" if test_ctx else "Function"
        add_node(
            G,
            f"{node_type.lower()}:{node_id}",
            node_type=node_type,
            symbol_id=symbol_id,
            module=module,
            name=name,
            path=path,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            num_calls=num_calls,
            decorators=decorators
        )

        # Index for resolution of CALLS
        if node_type == "Function":
            fn_index[(module, name)] = f"{node_type.lower()}:{node_id}"
            all_fn_names_in_module.setdefault(module, set()).add(name)
        else:
            # Test nodes也参与索引，避免无法被 CALLS 指向（有些测试也会被调用）
            fn_index[(module, name)] = f"{node_type.lower()}:{node_id}"
            all_fn_names_in_module.setdefault(module, set()).add(name)

        # Link to module/file
        add_node(G, f"module:{module}", node_type="Module", name=module)
        add_edge(G, f"{node_type.lower()}:{node_id}", f"module:{module}", etype="IN_MODULE")
        add_node(G, f"file:{path}", node_type="File", path=path, module=module,
                 has_test_flag=file_test_flag.get(path, False))
        add_edge(G, f"file:{path}", f"{node_type.lower()}:{node_id}", etype="DEFINES")
        add_edge(G, f"{node_type.lower()}:{node_id}", f"file:{path}", etype="IN_FILE")

    # Second pass: add CALLS edges, and DECORATED_BY for decorators/route
    for sym in symbols:
        kind = sym.get("kind")
        if kind not in ("Function", "Class"):
            continue

        symbol_id = sym.get("symbol_id")
        path = sym.get("path", "")
        module = sym.get("module", "")
        name = sym.get("name", "")
        decorators = sym.get("decorators") or []
        calls = sym.get("calls") or []

        src_is_test = is_test_context(
            path=path,
            module=module,
            has_test_flag=file_test_flag.get(path, False),
            fn_name=name
        )
        src_type = "Test" if src_is_test else "Function"
        src_id = f"{src_type.lower()}:{function_node_id(symbol_id, module, name)}"

        # CALLS: try to resolve within same module first (exact name match).
        for callee in calls:
            # Sometimes "route" appears as a call because of decorator usage in AST.
            if callee == "route":
                add_node(G, "ext:route", node_type="ExternalAPI", name="route")
                add_edge(G, src_id, "ext:route", etype="DECORATED_BY")
                continue

            # resolve to known functions in same module
            tgt_id = None
            if (module, callee) in fn_index:
                tgt_id = fn_index[(module, callee)]
            else:
                # if not found, treat as ExternalAPI / unresolved symbol
                add_node(G, f"ext:{callee}", node_type="ExternalAPI", name=callee)
                tgt_id = f"ext:{callee}"

            add_edge(G, src_id, tgt_id, etype="CALLS")

        # DECORATED_BY via decorators array (best-effort)
        for deco in decorators:
            if not deco:
                continue
            deco_name = str(deco)
            add_node(G, f"ext:{deco_name}", node_type="ExternalAPI", name=deco_name)
            add_edge(G, src_id, f"ext:{deco_name}", etype="DECORATED_BY")

    return G


def export_graph(G: nx.DiGraph, out_prefix: str) -> None:
    import pickle

    # Pickle
    pkl_path = f"{out_prefix}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[OK] Saved {pkl_path}")

    # GraphML
    graphml_path = f"{out_prefix}.graphml"
    try:
        nx.write_graphml(G, graphml_path)
        print(f"[OK] Saved {graphml_path}")
    except Exception as e:
        print(f"[WARN] GraphML export failed: {e}")

    # Flat JSON (nodes/edges)
    def _node_to_dict(nid: str, attrs: Dict[str, Any]):
        d = {"id": nid}
        d.update(attrs)
        # Normalize sets for JSON
        for k, v in list(d.items()):
            if isinstance(v, set):
                d[k] = sorted(list(v))
        return d

    def _edge_to_dict(s: str, t: str, attrs: Dict[str, Any]):
        d = {"source": s, "target": t}
        d.update(attrs)
        if isinstance(d.get("etype"), set):
            d["etype"] = sorted(list(d["etype"]))
        return d

    json_path = f"{out_prefix}.json"
    data = {
        "directed": True,
        "nodes": [_node_to_dict(n, G.nodes[n]) for n in G.nodes()],
        "edges": [_edge_to_dict(s, t, G.edges[s, t]) for s, t in G.edges()]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {json_path}")


def main():
    repo_base = Path(__file__).resolve().parents[1]
    default_index = repo_base / "processed_data" / "flask" / "index"
    default_out = repo_base / "processed_data" / "flask" / "graph"

    ap = argparse.ArgumentParser()
    ap.add_argument("--imports", default=str(default_index / "imports.jsonl"), help="Path to imports.jsonl")
    ap.add_argument("--symbols", default=str(default_index / "symbols.jsonl"), help="Path to symbols.jsonl")
    ap.add_argument("--out-prefix", default=str(default_out), help="Output file prefix (without extension)")
    args = ap.parse_args()

    G = build_graph(args.imports, args.symbols)
    # Basic stats
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    # Optional: quick sanity prints
    ntype_count = {}
    for n, d in G.nodes(data=True):
        ntype_count[d.get("node_type","UNK")] = ntype_count.get(d.get("node_type","UNK"), 0) + 1
    print("Node type counts:", ntype_count)

    etype_count = {}
    for s, t, d in G.edges(data=True):
        etypes = d.get("etype")
        if isinstance(etypes, set):
            for e in etypes:
                etype_count[e] = etype_count.get(e, 0) + 1
        else:
            etype_count[etypes] = etype_count.get(etypes, 0) + 1
    print("Edge type counts:", etype_count)

    export_graph(G, args.out_prefix)


if __name__ == "__main__":
    main()
