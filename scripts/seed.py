"""Ingest every corpus/*.md file through the deployed API.

Usage (PowerShell or bash):
    API_URL=https://xxxx.execute-api.us-east-1.amazonaws.com/v1 API_KEY=... python scripts/seed.py

The doc_id is the file name without extension, so re-running replaces documents in place.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
TIMEOUT_SECONDS = 60


def main() -> int:
    base = os.environ.get("API_URL", "").rstrip("/")
    key = os.environ.get("API_KEY", "")
    if not base or not key:
        print("Set API_URL and API_KEY first.", file=sys.stderr)
        return 2

    failures = 0
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() or path.stem
        body = json.dumps({"doc_id": path.stem, "title": title, "text": text}).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - URL is supplied by the operator
            f"{base}/documents",
            data=body,
            headers={"content-type": "application/json", "x-api-key": key},
            method="POST",
        )
        try:
            # nosec B310 - the operator supplies API_URL; only https endpoints are expected.
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310  # nosec B310
                print(f"{path.name}: {response.status} {response.read().decode('utf-8')}")
        except urllib.error.HTTPError as exc:
            failures += 1
            print(
                f"{path.name}: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')}",
                file=sys.stderr,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
