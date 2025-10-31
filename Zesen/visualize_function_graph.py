#!/usr/bin/env python3
"""
Generate a function-level knowledge graph (call graph) for a Python repo.

Usage:
    python visualize_function_graph.py /path/to/repo
        [--output path.html] [--reindex]
        [--layout sequential|radial|force] [--direction LR|RL|UD|DU]

The script relies on the symbol index produced by utils/index_repo.py. If the
index is missing or --reindex is set, it will rebuild the index automatically.
The resulting HTML (built with PyVis) visualizes functions/classes as nodes
and call relationships as directed edges.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Ensure project root (script parent) is on sys.path for absolute imports.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pyvis.network import Network
except ImportError as exc:  # pragma: no cover - graceful runtime check
    raise SystemExit(
        "Missing dependency 'pyvis'. Install it via 'pip install pyvis' and retry."
    ) from exc

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit("Missing dependency 'networkx'. Install it via 'pip install networkx'.") from exc

try:
    from utils.index_repo import build_index
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Cannot import utils.index_repo. Ensure you're running from the project context."
    ) from exc


Symbol = Dict[str, object]


def ensure_index(repo_root: Path, force: bool = False) -> Path:
    """Ensure the AST index exists; optionally rebuild."""
    index_dir = repo_root / "_index"
    symbols_file = index_dir / "symbols.jsonl"
    imports_file = index_dir / "imports.jsonl"

    if force or not (symbols_file.exists() and imports_file.exists()):
        print("ℹ️  Building symbol index ...")
        build_index(str(repo_root))
    else:
        print("ℹ️  Reusing existing symbol index.")

    if not symbols_file.exists():
        raise FileNotFoundError(f"Symbol index missing: {symbols_file}")
    return index_dir


def load_symbols(symbols_path: Path) -> List[Symbol]:
    """Load JSONL symbol data into memory."""
    symbols: List[Symbol] = []
    with symbols_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                symbols.append(json.loads(line))
            except json.JSONDecodeError as err:
                print(f"⚠️  Skipping malformed JSON line: {err}")
    print(f"ℹ️  Loaded {len(symbols)} symbols from {symbols_path}.")
    return symbols


def build_symbol_index(symbols: Iterable[Symbol]) -> Tuple[Dict[str, Symbol], Dict[str, List[str]]]:
    """Create helper lookups for symbol metadata and name-based resolution."""
    symbol_map: Dict[str, Symbol] = {}
    name_index: Dict[str, List[str]] = defaultdict(list)
    for sym in symbols:
        symbol_id = sym["symbol_id"]
        symbol_map[str(symbol_id)] = sym
        name_index[str(sym["name"])].append(str(symbol_id))
    return symbol_map, name_index


def resolve_callee(
    caller_id: str,
    callee_name: str,
    symbol_map: Dict[str, Symbol],
    name_index: Dict[str, List[str]],
) -> Optional[str]:
    """Resolve a callee name to a unique symbol_id when possible."""
    candidates = name_index.get(callee_name, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    caller_mod = str(symbol_map[caller_id]["module"])

    same_module = [cid for cid in candidates if symbol_map[cid]["module"] == caller_mod]
    if len(same_module) == 1:
        return same_module[0]

    caller_pkg = caller_mod.split(".")[0]
    same_package = [
        cid for cid in candidates if str(symbol_map[cid]["module"]).split(".")[0] == caller_pkg
    ]
    if len(same_package) == 1:
        return same_package[0]

    return None


def build_call_edges(symbol_map: Dict[str, Symbol], name_index: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Create directed edges (caller -> callee) based on symbol call metadata."""
    edges: List[Tuple[str, str]] = []
    unresolved = 0
    for symbol_id, sym in symbol_map.items():
        calls = sym.get("calls") or []
        for callee_name in calls:
            callee_id = resolve_callee(symbol_id, callee_name, symbol_map, name_index)
            if callee_id:
                if callee_id != symbol_id:
                    edges.append((symbol_id, callee_id))
            else:
                unresolved += 1
    print(f"ℹ️  Constructed {len(edges)} edges. Unresolved call targets: {unresolved}.")
    return edges


def module_color(module_name: str) -> str:
    """Derive a consistent color per module."""
    palette = [
        "#5B8FF9",
        "#5AD8A6",
        "#5D7092",
        "#F6BD16",
        "#E8684A",
        "#6DC8EC",
        "#9270CA",
        "#FF9D4D",
        "#269A99",
    ]
    idx = abs(hash(module_name)) % len(palette)
    return palette[idx]


def compute_sequential_layout(
    symbol_ids: Iterable[str],
    edges: Iterable[Tuple[str, str]],
    direction: str,
) -> Tuple[Dict[str, Tuple[float, float]], set[str], set[str]]:
    """Produce deterministic positions using condensation DAG topology."""
    graph = nx.DiGraph()
    graph.add_nodes_from(symbol_ids)
    graph.add_edges_from(edges)

    if graph.number_of_nodes() == 0:
        return {}, set(), set()

    sccs = list(nx.strongly_connected_components(graph))
    component_index: Dict[str, int] = {}
    for idx, component in enumerate(sccs):
        for node in component:
            component_index[node] = idx

    condensation = nx.DiGraph()
    condensation.add_nodes_from(range(len(sccs)))
    for u, v in graph.edges():
        cu, cv = component_index[u], component_index[v]
        if cu != cv:
            condensation.add_edge(cu, cv)

    component_level: Dict[int, int] = {}
    try:
        for level, layer in enumerate(nx.topological_generations(condensation)):
            for comp_id in layer:
                component_level[comp_id] = level
    except nx.NetworkXError:
        order = list(nx.topological_sort(condensation))
        component_level = {comp_id: idx for idx, comp_id in enumerate(order)}

    level_nodes: Dict[int, List[str]] = defaultdict(list)
    for node in graph.nodes():
        comp_id = component_index[node]
        level = component_level.get(comp_id, 0)
        level_nodes[level].append(node)

    spacing_major = 230.0
    spacing_minor = 75.0
    positions: Dict[str, Tuple[float, float]] = {}

    for level, nodes in level_nodes.items():
        nodes_sorted = sorted(nodes)
        for idx, node in enumerate(nodes_sorted):
            offset = idx - (len(nodes_sorted) - 1) / 2
            if direction in ("LR", "RL"):
                x = level * spacing_major
                y = offset * spacing_minor
                if direction == "RL":
                    x = -x
            else:
                x = offset * spacing_minor
                y = level * spacing_major
                if direction == "DU":
                    y = -y
            positions[node] = (x, y)

    sources = {
        node for node in graph.nodes() if graph.in_degree(node) == 0 and graph.out_degree(node) > 0
    }
    sinks = {
        node for node in graph.nodes() if graph.out_degree(node) == 0 and graph.in_degree(node) > 0
    }

    return positions, sources, sinks


def compute_radial_positions(symbol_ids: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    """Place nodes on concentric circles grouped by first letter of module."""
    positions: Dict[str, Tuple[float, float]] = {}
    grouped: Dict[str, List[str]] = defaultdict(list)
    for sid in symbol_ids:
        prefix = sid.split("::", 1)[0].split(".")[0]
        grouped[prefix].append(sid)

    radius_step = 220.0
    for layer_idx, (prefix, nodes) in enumerate(sorted(grouped.items())):
        radius = (layer_idx + 1) * radius_step
        count = len(nodes)
        for idx, node in enumerate(sorted(nodes)):
            angle = (2 * math.pi * idx) / count if count > 0 else 0.0
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[node] = (x, y)
    return positions


def build_visualization(
    symbols: Iterable[Symbol],
    edges: Iterable[Tuple[str, str]],
    output_path: Path,
    layout: str,
    direction: str,
) -> None:
    """Render the call graph using PyVis."""
    net = Network(height="750px", width="100%", directed=True, notebook=False)

    modules = [str(sym["module"]) for sym in symbols]
    module_levels = {module: idx for idx, module in enumerate(sorted(set(modules)))}

    symbol_ids = [str(sym["symbol_id"]) for sym in symbols]
    positions: Dict[str, Tuple[float, float]] = {}
    sources: set[str] = set()
    sinks: set[str] = set()

    if layout == "sequential":
        positions, sources, sinks = compute_sequential_layout(symbol_ids, edges, direction)
        options = {
            "physics": {"enabled": False},
            "interaction": {
                "dragNodes": True,
                "navigationButtons": True,
                "multiselect": True,
            },
        }
        net.set_options(json.dumps(options))
    elif layout == "radial":
        positions = compute_radial_positions(symbol_ids)
        options = {
            "physics": {"enabled": False},
            "interaction": {
                "dragNodes": True,
                "navigationButtons": True,
                "multiselect": True,
            },
        }
        net.set_options(json.dumps(options))
    elif layout == "force":
        net.barnes_hut(gravity=-2000, central_gravity=0.1, spring_length=120)

    for sym in symbols:
        symbol_id = str(sym["symbol_id"])
        label = f"{sym['name']}()"
        title_parts = [
            f"<b>{sym['symbol_id']}</b>",
            f"Module: {sym['module']}",
            f"File: {sym['path']}",
            f"Kind: {sym['kind']}",
        ]
        doc = sym.get("docstring")
        if doc:
            title_parts.append(f"Docstring: {doc[:300]}{'…' if len(doc) > 300 else ''}")
        node_kwargs = {
            "label": label,
            "title": "<br>".join(title_parts),
            "color": module_color(str(sym["module"])),
        }
        if layout == "sequential":
            if symbol_id not in positions:
                continue
            x, y = positions[symbol_id]
            node_kwargs["x"] = x
            node_kwargs["y"] = y
            node_kwargs["fixed"] = True
            if symbol_id in sources:
                node_kwargs["shape"] = "box"
                node_kwargs["borderWidth"] = 2
            elif symbol_id in sinks:
                node_kwargs["shape"] = "triangle"
                node_kwargs["borderWidth"] = 2
        elif layout == "radial":
            if symbol_id not in positions:
                continue
            x, y = positions[symbol_id]
            node_kwargs["x"] = x
            node_kwargs["y"] = y
            node_kwargs["fixed"] = True

        net.add_node(symbol_id, **node_kwargs)

    for caller, callee in edges:
        if layout in {"sequential", "radial"}:
            if caller not in positions or callee not in positions:
                continue
        net.add_edge(caller, callee)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)
    print(f"✅ Knowledge graph written to {output_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize function dependencies as a knowledge graph.")
    parser.add_argument(
        "repo",
        type=Path,
        help="Path to the repository root (must contain Python files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <repo>/_index/function_graph.html).",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild the symbol index before visualization.",
    )
    parser.add_argument(
        "--layout",
        choices=["sequential", "radial", "force"],
        default="sequential",
        help="Graph layout strategy (default: sequential condensation layout).",
    )
    parser.add_argument(
        "--direction",
        choices=["LR", "RL", "UD", "DU"],
        default="LR",
        help="Flow direction for sequential layout (default: LR; ignored by other layouts).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo.resolve()
    if not repo_root.exists():
        raise SystemExit(f"Repository path not found: {repo_root}")

    index_dir = ensure_index(repo_root, force=args.reindex)
    symbols_path = index_dir / "symbols.jsonl"
    symbols = load_symbols(symbols_path)
    if not symbols:
        raise SystemExit("No symbols found; ensure the repo contains Python files.")

    symbol_map, name_index = build_symbol_index(symbols)
    edges = build_call_edges(symbol_map, name_index)
    if not edges:
        print("⚠️  No call relationships detected; visualization may be sparse.")

    output_path = args.output or (index_dir / "function_graph.html")
    build_visualization(symbols, edges, output_path, layout=args.layout, direction=args.direction)


if __name__ == "__main__":
    main(sys.argv[1:])
