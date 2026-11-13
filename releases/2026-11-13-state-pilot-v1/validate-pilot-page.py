"""Verify local report assets and same-document anchors."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.references = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            if values["id"] in self.ids:
                raise ValueError(f"Duplicate HTML id: {values['id']}")
            self.ids.add(values["id"])
        for key in ("href", "src"):
            if key in values:
                self.references.append(values[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    args = parser.parse_args()
    page = args.page.resolve()
    report = ReportParser()
    report.feed((page / "index.html").read_text(encoding="utf-8"))
    for reference in report.references:
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc:
            continue
        if not parsed.path and parsed.fragment:
            if parsed.fragment not in report.ids:
                raise ValueError(f"Missing anchor: {reference}")
            continue
        if not parsed.path:
            continue
        target = (page / unquote(parsed.path)).resolve()
        if not target.exists():
            raise ValueError(f"Missing local report link: {reference}")
    print(f"Checked {len(report.references)} report references and {len(report.ids)} ids.")


if __name__ == "__main__":
    main()
