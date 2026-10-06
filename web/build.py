"""Build the web game as a static site, to play in any browser without a server.

    python -m web.build [out]    # default: _site

The site is the page and `wist.zip`, the Python it runs: the engine, the trained bots and
`web/server.py`. Finding no server, the page loads Pyodide (Python built for the browser)
and NumPy from a CDN and plays with `server.in_browser()`. GitHub Pages serves it
(.github/workflows/pages.yml); `python -m http.server -d _site` serves it locally.

The page asks for the archive by a hash of it (`wist.zip?v=...`): browsers keep both for a
while, and a newer page must never run an archive cached from an older deploy.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    *sorted(ROOT.glob("danish_wist/*.py")),
    *(ROOT / "learn" / name for name in ("__init__.py", "encoding.py", "inference.py")),
    *(ROOT / "web" / name for name in ("__init__.py", "server.py", "rl-005.npz", "rl-006.npz")),
]


FETCH = 'fetch("wist.zip")'  # how the page asks for the archive, given its hash here


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "wist.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in FILES:
            archive.write(path, path.relative_to(ROOT).as_posix())
    page = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    if page.count(FETCH) != 1:
        raise ValueError(f"web/index.html should ask for the archive once, as {FETCH}")
    version = hashlib.sha256((out / "wist.zip").read_bytes()).hexdigest()[:12]
    page = page.replace(FETCH, f'fetch("wist.zip?v={version}")')
    (out / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
