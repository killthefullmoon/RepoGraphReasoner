from pyvis.network import Network
import pickle
import networkx as nx
import os
import pyvis
from pathlib import Path

template_path = os.path.join(os.path.dirname(pyvis.__file__), "templates")
repo_base = Path(__file__).resolve().parents[1]
graph_path = repo_base / "processed_data" / "flask" / "graph.pkl"
out_html = repo_base / "processed_data" / "flask" / "graph_visualization.html"

# Load graph
with open(graph_path, "rb") as f:
    G = pickle.load(f)

H = G

net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff")

# 给不同节点上色
colors = {
    "Module": "#ffcc00",
    "Function": "#66cc66",
    "Test": "#ff6666",
    "Library": "#66b2ff",
    "ExternalAPI": "#cc99ff",
    "File": "#cccccc"
}

for n, data in H.nodes(data=True):
    label = data.get("name", n)
    ntype = data.get("node_type", "Other")
    color = colors.get(ntype, "#dddddd")
    net.add_node(n, label=label, title=str(data), color=color, shape="dot")

for s, t, data in H.edges(data=True):
    etype = ", ".join(data.get("etype", [])) if isinstance(data.get("etype"), set) else data.get("etype")
    net.add_edge(s, t, title=etype, color="#999999", arrows="to")


# Export to HTML
net.write_html(str(out_html), open_browser=False)
print(f"✅ Visualization saved to {out_html}")
