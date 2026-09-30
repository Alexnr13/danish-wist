"""Do the checkpoints' differences repeat on other deals? (TRAINING.md, "The recipe for a long run")

    python results/rl-006/recipe-review/replicate.py

rl-006's checkpoints 5810 to 6000, each paired with their weight average, were measured against
RuleBot on two independent sets of deals (seeds 31 and 32, then 33 and 34). If the checkpoints
differed only by the luck of the deals, the two measurements would not agree; if the weights
really differ, they would, up to each value's own noise.
"""

import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["average-and-its-members-vs-rule.txt", "average-and-its-members-vs-rule-seeds-33-34.txt"]


def members(name: str) -> list[tuple[float, float]]:
    paired = (HERE / name).read_text().split("paired with")[1]
    found = re.finditer(r"policy-\d+\.pt: ([+-][\d.]+) ± ([\d.]+)", paired)
    return [(float(m[1]), float(m[2])) for m in found]


a, b = (members(name) for name in FILES)
x, y = [v for v, _ in a], [v for v, _ in b]
r = statistics.correlation(x, y)
noise = statistics.mean(ci for _, ci in a + b) / 1.96  # the sd of one paired value
spread = (r * statistics.stdev(x) * statistics.stdev(y)) ** 0.5  # what the two share
print(f"{len(x)} checkpoints, paired with their average against RuleBot")
print(f"mean: {statistics.mean(x):+.1f} (seeds 31, 32), {statistics.mean(y):+.1f} (seeds 33, 34)")
print(f"correlation of the two measurements: {r:.2f}")
print(f"spread of the checkpoints themselves: sd {spread:.1f}; noise of one value: sd {noise:.1f}")
pairs = list(zip(x, y, strict=True))
below, above = sum(u < 0 and v < 0 for u, v in pairs), sum(u > 0 and v > 0 for u, v in pairs)
print(f"below the average on both: {below}; above on both: {above}")
