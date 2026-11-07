"""Token use from the API-reported counts stored with each cached reply.
python cost_report.py --out outputs [--shows-per-week 5] [--all]
Set PRICE_IN_PER_M / PRICE_OUT_PER_M (USD per million tokens) to also see money."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--shows-per-week", type=float, default=5, help="ASSUMPTION: how many new 50-episode shows per week")
    ap.add_argument("--eps-per-show", type=int, default=50)
    ap.add_argument("--all", action="store_true", help="include every provider/model in the cache")
    a = ap.parse_args()
    prov = config.provider()
    mod = config.model(prov)
    per_ep, per_task = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])
    providers = set()
    for f in (Path(a.out) / "cache").glob("*.json"):
        try:
            c = json.loads(f.read_text())
        except ValueError:
            continue
        if not a.all and (c["provider"] != prov or c["model"] != mod):
            continue
        providers.add(f"{c['provider']}/{c['model']}")
        u = c.get("usage", {})
        for tab, k in ((per_ep, c.get("episode")), (per_task, c["task"])):
            tab[k][0] += u.get("in", 0)
            tab[k][1] += u.get("out", 0)
            tab[k][2] += 1
    if not per_ep:
        raise SystemExit("No cached replies found for this provider/model. Run run_all.py first (or pass --all).")
    n = len(per_ep)
    tin = sum(v[0] for v in per_ep.values())
    tout = sum(v[1] for v in per_ep.values())
    calls = sum(v[2] for v in per_ep.values())
    ain, aout, acalls = tin / n, tout / n, calls / n
    est = " (ESTIMATED from text length - mock provider)" if "mock/mock" in providers else ""
    print(f"Models in this report: {', '.join(sorted(providers))}{est}")
    print(f"Episodes measured: {n}   API calls: {calls}\n")
    print("Tokens per episode (avg):  in {:,.0f}   out {:,.0f}   calls {:.1f}".format(ain, aout, acalls))
    print("By step (avg per episode):")
    for t, v in sorted(per_task.items()):
        print(f"  {t:8s} in {v[0]/n:>8,.0f}   out {v[1]/n:>7,.0f}   calls {v[2]/n:.1f}")
    show_in, show_out = ain * a.eps_per_show, aout * a.eps_per_show
    wk_in, wk_out = show_in * a.shows_per_week, show_out * a.shows_per_week
    print(f"\nPer {a.eps_per_show}-episode show:    in {show_in:,.0f}   out {show_out:,.0f}   calls {acalls*a.eps_per_show:,.0f}")
    print(f"Per week ({a.shows_per_week:g} new shows, assumed): in {wk_in:,.0f}   out {wk_out:,.0f}   calls {acalls*a.eps_per_show*a.shows_per_week:,.0f}")
    pi, po = os.getenv("PRICE_IN_PER_M"), os.getenv("PRICE_OUT_PER_M")
    if pi and po:
        cost = lambda i, o: i / 1e6 * float(pi) + o / 1e6 * float(po)
        print(f"\nMoney at ${pi}/M in, ${po}/M out:  per episode ${cost(ain, aout):.4f}   per show ${cost(show_in, show_out):.2f}   per week ${cost(wk_in, wk_out):.2f}")
    else:
        print("\n(Set PRICE_IN_PER_M and PRICE_OUT_PER_M to see dollar cost; prices are not hard-coded.)")


if __name__ == "__main__":
    main()
