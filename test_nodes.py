from src.nodes.automation_nodes import get_all_nodes_by_category

cats = get_all_nodes_by_category()
print(f"Total categories: {len(cats)}")
for cat, nodes in cats.items():
    display_names = [n.display_name for n in nodes[:2]]
    print(f"  {cat}: {len(nodes)} nodes - {display_names}")
