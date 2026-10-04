"""Build the web game as a static site, to play in any browser without a server.

    python -m web.build [out]    # default: _site

The site is the page and `wist.zip`, the Python it runs: the engine, the trained bot and
`web/server.py`. Finding no server, the page loads Pyodide (Python built for the browser)
and NumPy from a CDN and plays with `server.in_browser()`. GitHub Pages serves it
(.github/workflows/pages.yml); `python -m http.server -d _site` serves it locally.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    *sorted(ROOT.glob("danish_wist/*.py")),
    *(ROOT / "learn" / name for name in ("__init__.py", "encoding.py", "inference.py")),
    *(ROOT / "web" / name for name in ("__init__.py", "server.py", "bot.npz")),
]


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_bytes((ROOT / "web" / "index.html").read_bytes())
    with zipfile.ZipFile(out / "wist.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in FILES:
            archive.write(path, path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
