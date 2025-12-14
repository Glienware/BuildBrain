"""Catalog of node categories used by the AI agent workspace."""

def _slugify(node_label: str) -> str:
    slug = node_label.lower()
    for ch in ['/', ' ', '(', ')', ',', '\'', '"']:
        slug = slug.replace(ch, '_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug.strip('_')


def _build_nodes(labels: list[str]) -> list[dict]:
    return [
        {
            "label": label,
            "node_type": _slugify(label)
        }
        for label in labels
    ]


NODE_CATEGORIES = []  # Catalog disabled while automation is redesigned
