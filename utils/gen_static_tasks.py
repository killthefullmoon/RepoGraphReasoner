import json, os
from collections import defaultdict

# ========== Configuration ==========
REPO_ROOT = "../data/repos/flask"
INDEX_DIR = os.path.join(REPO_ROOT, "_index")
QA_OUT = os.path.join(REPO_ROOT, "qa/reasoning_static.jsonl")
os.makedirs(os.path.dirname(QA_OUT), exist_ok=True)

SYMBOLS_FILE = os.path.join(INDEX_DIR, "symbols.jsonl")
IMPORTS_FILE = os.path.join(INDEX_DIR, "imports.jsonl")

# ========== Helpers ==========
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ========== Load data ==========
print("📖 Loading symbol and import indexes...")
symbols = load_jsonl(SYMBOLS_FILE)
imports = load_jsonl(IMPORTS_FILE)
print(f"Loaded {len(symbols)} symbols and {len(imports)} files.")

# Build quick lookup maps
symbol_map = {s["symbol_id"]: s for s in symbols}
symbol_index = defaultdict(list)
for s in symbols:
    symbol_index[s["name"]].append(s["symbol_id"])

# ========== Build call graph ==========
print("🧩 Building call graph...")
call_graph = defaultdict(list)
for s in symbols:
    caller = s["symbol_id"]
    for callee_name in s.get("calls", []):
        if callee_name in symbol_index and len(symbol_index[callee_name]) == 1:
            callee = symbol_index[callee_name][0]
            if callee != caller:
                call_graph[caller].append(callee)
print(f"Built call graph with {len(call_graph)} caller nodes.")

# ========== Utility: find unique terminal chain ==========
def find_unique_terminal(caller, graph, max_depth=5):
    visited, stack = set(), [(caller, [caller])]
    terminals = []
    while stack:
        cur, path = stack.pop()
        if cur not in graph or not graph[cur]:
            terminals.append(path)
        else:
            for nxt in graph[cur]:
                if nxt not in visited and len(path) < max_depth:
                    visited.add(nxt)
                    stack.append((nxt, path + [nxt]))
    if len(terminals) == 1:
        return terminals[0]
    return None

# ========== 1️⃣ Function call chain reasoning ==========
qa_tasks = []
for s in symbols:
    chain = find_unique_terminal(s["symbol_id"], call_graph)
    if chain and len(chain) > 1:
        start, end = symbol_map[chain[0]], symbol_map[chain[-1]]
        qa_tasks.append({
            "type": "call_chain_reasoning",
            "question": f"When calling `{start['name']}`, which function is ultimately invoked?",
            "answer": end["name"],
            "ground_truth": end["symbol_id"],
            "reasoning_path": [symbol_map[n]["name"] for n in chain],
            "evidence": {"start_path": start["path"], "end_path": end["path"]},
            "verified": True
        })

# ========== 2️⃣ Cross-module reasoning ==========
for caller, callees in call_graph.items():
    caller_mod = symbol_map[caller]["module"]
    target_modules = set(
        symbol_map[c]["module"]
        for c in callees
        if symbol_map[c]["module"] != caller_mod
    )
    if len(target_modules) == 1:
        qa_tasks.append({
            "type": "cross_module_reasoning",
            "question": f"Which module's function does `{symbol_map[caller]['name']}` call?",
            "answer": list(target_modules)[0],
            "ground_truth": list(target_modules)[0],
            "evidence": {"path": symbol_map[caller]["path"]},
            "verified": True
        })

# ========== 3️⃣ Import-call reasoning ==========
module_to_imports = {f["module"]: f.get("imports", []) for f in imports}
for s in symbols:
    src_mod = s["module"]
    imported_mods = module_to_imports.get(src_mod, [])
    callees = call_graph.get(s["symbol_id"], [])
    for imp_mod in imported_mods:
        targets = [
            symbol_map[c]["name"]
            for c in callees
            if symbol_map[c]["module"].startswith(imp_mod)
        ]
        targets = unique(targets)
        if len(targets) == 1:
            qa_tasks.append({
                "type": "import_call_reasoning",
                "question": f"In module `{src_mod}`, which function from imported module `{imp_mod}` is called?",
                "answer": targets[0],
                "ground_truth": targets[0],
                "evidence": {"path": s["path"], "import": imp_mod},
                "verified": True
            })

# ========== 4️⃣ Class-method dependency reasoning ==========
for s in symbols:
    if s["kind"] == "Function" and s.get("parent"):
        class_a = s["parent"]
        callees = call_graph.get(s["symbol_id"], [])
        class_targets = set()
        for callee in callees:
            callee_obj = symbol_map[callee]
            if callee_obj.get("parent") and callee_obj["parent"] != class_a:
                class_targets.add(f"{callee_obj['parent']}.{callee_obj['name']}")
        if len(class_targets) == 1:
            qa_tasks.append({
                "type": "class_method_dependency",
                "question": f"Which class method does `{class_a}.{s['name']}` depend on?",
                "answer": list(class_targets)[0],
                "ground_truth": list(class_targets)[0],
                "evidence": {"path": s["path"], "parent": class_a},
                "verified": True
            })

# ========== 5️⃣ Test coverage reasoning ==========
test_files = [f for f in imports if f.get("has_test_flag")]
for test_file in test_files:
    test_mod = test_file["module"]
    for imp in test_file.get("imports", []):
        funcs = [
            s for s in symbols
            if s["module"] == test_mod and s["name"].startswith("test")
        ]
        impl_funcs = [
            s for s in symbols if s["module"].startswith(imp)
        ]
        for t in funcs:
            for impl in impl_funcs:
                if impl["name"] in t.get("calls", []) and impl["name"] != t["name"]:
                    qa_tasks.append({
                        "type": "test_coverage_reasoning",
                        "question": f"Which test function verifies the behavior of `{impl['name']}`?",
                        "answer": t["name"],
                        "ground_truth": t["symbol_id"],
                        "evidence": {"test_path": t["path"], "impl_path": impl["path"]},
                        "verified": True
                    })

# ========== 6️⃣ Import chain (multi-hop) reasoning ==========
module_imports = {f["module"]: f.get("imports", []) for f in imports}

def find_unique_import_chain(start_mod, max_depth=3):
    visited, stack = set(), [(start_mod, [start_mod])]
    terminals = []
    while stack:
        mod, path = stack.pop()
        if mod not in module_imports or not module_imports[mod]:
            terminals.append(path)
        else:
            for nxt in module_imports[mod]:
                if nxt not in visited and len(path) < max_depth:
                    visited.add(nxt)
                    stack.append((nxt, path + [nxt]))
    if len(terminals) == 1:
        return terminals[0]
    return None

for mod in module_imports:
    chain = find_unique_import_chain(mod)
    if chain and len(chain) > 2:
        qa_tasks.append({
            "type": "import_chain_reasoning",
            "question": f"Through which modules does `{chain[0]}` indirectly import `{chain[-1]}`?",
            "answer": " → ".join(chain[1:]),
            "ground_truth": chain[-1],
            "reasoning_path": chain,
            "verified": True
        })

# ========== Deduplicate & Save ==========
def deduplicate(tasks):
    seen, out = set(), []
    for t in tasks:
        key = (t["question"], t["answer"])
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out

qa_tasks = deduplicate(qa_tasks)
write_jsonl(QA_OUT, qa_tasks)
print(f"✅ Generated {len(qa_tasks)} static reasoning tasks.")
print(f"📦 Saved to {QA_OUT}")
