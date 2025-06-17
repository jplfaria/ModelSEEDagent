import re
from pathlib import Path

docs_root = Path(__file__).resolve().parent.parent / "docs"

link_map = {
    r"user/README\.md": "getting-started/quickstart-cli.md",
    r"user/INTERACTIVE_GUIDE\.md": "getting-started/interactive-guide.md",
}

poetry_patterns = [
    (re.compile(r"poetry install[^\n]*"), "pip install -e ."),
    (re.compile(r"poetry build[^\n]*"), "python -m build"),
]


def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text
    # replace links
    for pat, repl in link_map.items():
        text = re.sub(pat, repl, text)
    # replace poetry cmds
    for pat, repl in poetry_patterns:
        text = pat.sub(repl, text)
    # add draft front matter for archive/development
    if "docs/archive" in str(path) or "/development/" in str(path):
        if not text.lstrip().startswith("---"):
            text = f"---\ndraft: true\n---\n\n" + text
        elif "draft:" not in text.split("---", 2)[1]:
            parts = text.split("---")
            parts[1] += "\ndraft: true\n"
            text = "---".join(parts)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main():
    modified = 0
    for md_file in docs_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        if process_file(md_file):
            modified += 1
    print(f"Modified {modified} Markdown files")


if __name__ == "__main__":
    main()
