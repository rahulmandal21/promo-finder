"""Compare the tool's top 1 / top 3 per episode with YOUR gold.json by time overlap.
python evaluate.py --gold gold.json --out outputs
gold.json: [{"episode": 3, "start": "00:05.0", "end": "00:41.5"}, ...]   (seconds or mm:ss also fine)"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import config
from utils import overlap_ratio


def _t(x):
    if isinstance(x, (int, float)):
        return float(x)
    m = re.match(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$", str(x).strip())
    if not m:
        raise ValueError(f"bad time '{x}'")
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def load_gold(path):
    g = []
    for r in json.loads(Path(path).read_text()):
        g.append({"episode": int(r["episode"]), "segments": [{"episode": int(r["episode"]),
                  "start": _t(r["start"]), "end": _t(r["end"])}]})
    return g


def load_picks(out):
    d = json.loads((Path(out) / "promos_all.json").read_text())["promos"]
    picks = [p for p in d if p.get("picked_rank") is not None]
    by_ep = defaultdict(list)
    for p in sorted(picks, key=lambda x: x["picked_rank"]):
        by_ep[p["segments"][0]["episode"]].append(p)
    return by_ep


def hit(p, g):
    return overlap_ratio(p, g) >= config.HIT_OVERLAP


def evaluate(gold, by_ep):
    gold_eps = sorted({g["episode"] for g in gold})
    t1 = t3 = 0
    per_src, per_shape = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    covered = 0
    lines = []
    for ep in gold_eps:
        gs = [g for g in gold if g["episode"] == ep]
        ps = sorted(by_ep.get(ep, []), key=lambda x: x["picked_rank"])[:3]
        h = [any(hit(p, g) for g in gs) for p in ps]
        t1 += bool(h and h[0])
        t3 += any(h)
        covered += sum(any(hit(p, g) for p in ps) for g in gs)
        for p, ok in zip(ps, h):
            for tab, k in ((per_src, p["source"]), (per_shape, p["shape"])):
                tab[k][0] += ok
                tab[k][1] += 1
        lines.append(f"- Ep{ep}: top1 {'HIT' if h and h[0] else 'miss'}, top3 {'HIT' if any(h) else 'miss'} "
                     f"({len(ps)} picks vs {len(gs)} gold)")
    n = len(gold_eps)
    rate = lambda a, b: f"{a}/{b} = {a / b:.0%}" if b else "n/a"
    out = [f"# Evaluation vs gold ({n} episodes with gold, {len(gold)} gold promos)",
           f"Hit = overlap of at least {config.HIT_OVERLAP:.0%} of the shorter clip.\n",
           f"- Top-1 hit rate: {rate(t1, n)}", f"- Top-3 hit rate: {rate(t3, n)}",
           f"- Gold promos covered by a top-3 pick: {rate(covered, len(gold))}",
           "\n## Precision of top-3 picks by source (picks that hit a gold promo)"]
    out += [f"- {k}: {rate(*v)}" for k, v in sorted(per_src.items())]
    out.append("\n## ...by shape")
    out += [f"- {k}: {rate(*v)}" for k, v in sorted(per_shape.items())]
    out.append("\n## Per episode")
    return "\n".join(out + lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold.json")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    if not Path(a.gold).exists():
        raise SystemExit(f"{a.gold} not found. Copy gold.example.json to gold.json and fill in your own best promos.")
    rep = evaluate(load_gold(a.gold), load_picks(a.out))
    print(rep)
    (Path(a.out) / "evaluation.md").write_text(rep)
