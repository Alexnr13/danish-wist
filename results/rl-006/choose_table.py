"""The choosing table from `results/rl-005/choose.sh`'s output: each candidate's card play and
full game against each field, paired with the start, and the smallest of the fields.

    python results/rl-006/choose_table.py results/rl-006/choose-18000-a.txt [more parts ...]

Parts of one choice (the same start, seeds and fields, other candidates) are read together.
"""

import re
import sys

SECTION = re.compile(r"^== (.*)$", re.M)
PAIRED = re.compile(r"^  (?:play:)?(\S+): ([+-][\d.]+) ± ([\d.]+)$", re.M)
OWN = re.compile(
    r"^(?:play:)?(\S+) against a field.*\nadvantage per deal: ([+-][\d.]+) ± ([\d.]+)", re.M
)


def read(paths: list[str]) -> tuple[list[str], dict, dict]:
    """The sections in order, and per section each candidate's paired and own results."""
    sections: list[str] = []
    paired: dict[str, dict[str, tuple[float, float]]] = {}
    own: dict[str, dict[str, tuple[float, float]]] = {}
    for path in paths:
        text = open(path).read()
        heads = list(SECTION.finditer(text))
        for head, after in zip(heads, heads[1:] + [None], strict=True):
            body = text[head.end() : after.start() if after else len(text)]
            name = head[1]
            if name not in paired:
                sections.append(name)
                paired[name], own[name] = {}, {}
            paired[name] |= {m[1]: (float(m[2]), float(m[3])) for m in PAIRED.finditer(body)}
            own[name] |= {m[1]: (float(m[2]), float(m[3])) for m in OWN.finditer(body)}
    return sections, paired, own


def short(name: str) -> str:
    found = re.search(r"(rl-00\d)/checkpoints/(policy|magnet)-0*(\d+)", name)
    if found:
        return f"{found[1]}'s {'magnet ' * (found[2] == 'magnet')}{found[3]}"
    return "RuleBot" if name == "rule" else name.removeprefix("runs/").removesuffix("/policy.pt")


def main() -> None:
    sections, paired, own = read(sys.argv[1:])
    fields = [s.removeprefix("the full game against a field of ") for s in sections[1:]]
    start = next(n for n in own[sections[0]] if n not in paired[sections[0]])
    names = ["card play", *(f"vs {short(f)}" for f in fields)]
    results = [own[s][start] for s in sections]
    print(
        f"{short(start)} itself:",
        ", ".join(f"{n} {r[0]:+.1f} ± {r[1]:.1f}" for n, r in zip(names, results, strict=True)),
    )
    print("| Candidate | " + " | ".join(n[0].upper() + n[1:] for n in names) + " | Smallest |")
    print("|---" * (len(fields) + 3) + "|")
    for name in paired[sections[0]]:
        cells = [paired[s].get(name) for s in sections]
        smallest = min(c[0] for c in cells[1:] if c)
        print(
            f"| {short(name)} | "
            + " | ".join(f"{c[0]:+.1f} ± {c[1]:.1f}" if c else "" for c in cells)
            + f" | {smallest:+.1f} |"
        )


if __name__ == "__main__":
    main()
