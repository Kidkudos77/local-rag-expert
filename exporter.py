"""
exporter.py — Export chat history and annotations to downloadable formats.
No disk writes — everything generated in memory for cloud compatibility.
"""

import io
from datetime import datetime


def export_as_markdown(messages: list, title: str = "RAG Expert Chat History") -> bytes:
    """Convert chat history to a formatted markdown document."""
    lines = [
        f"# {title}",
        f"*Exported: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*",
        "",
        "---",
        "",
    ]
    for msg in messages:
        if msg["role"] == "user":
            lines.append(f"### 🧑 You")
            lines.append(msg["content"])
        else:
            lines.append(f"### 🤖 Assistant")
            lines.append(msg["content"])
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).encode("utf-8")


def export_as_txt(messages: list, title: str = "RAG Expert Chat History") -> bytes:
    """Convert chat history to plain text."""
    lines = [
        title,
        f"Exported: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        "=" * 60,
        "",
    ]
    for msg in messages:
        role = "YOU" if msg["role"] == "user" else "ASSISTANT"
        lines.append(f"{role}:")
        lines.append(msg["content"])
        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines).encode("utf-8")


def export_annotations_as_markdown(annotations: list) -> bytes:
    """Export annotations to markdown."""
    lines = [
        "# Research Annotations",
        f"*Exported: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*",
        "",
        "---",
        "",
    ]
    if not annotations:
        lines.append("*No annotations yet.*")
    else:
        for i, ann in enumerate(annotations, 1):
            lines.append(f"## Annotation {i}")
            lines.append(f"**Document:** `{ann.get('doc', 'Unknown')}`")
            lines.append(f"**Added:** {ann.get('timestamp', 'Unknown')}")
            lines.append("")
            lines.append(f"**Passage:**")
            lines.append(f"> {ann.get('passage', '')}")
            lines.append("")
            lines.append(f"**Note:**")
            lines.append(ann.get("note", ""))
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines).encode("utf-8")
