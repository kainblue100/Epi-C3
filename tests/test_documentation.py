import argparse
import re
from pathlib import Path
from urllib.parse import unquote

from c3vcell.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_documentation_local_links_resolve() -> None:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    missing = []
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for link in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", content):
            target = link.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / unquote(target)).is_file():
                missing.append(f"{document.relative_to(ROOT)}: {link}")
    assert not missing, missing


def test_cli_reference_includes_every_command() -> None:
    parser = build_parser()
    subparsers = [
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers) == 1
    content = (ROOT / "docs" / "training-and-cli.zh-CN.md").read_text(encoding="utf-8")
    undocumented = sorted(name for name in subparsers[0].choices if f"`{name}`" not in content)
    assert not undocumented, undocumented
