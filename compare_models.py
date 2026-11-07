"""Run the same 5 episodes on two models and report how much their picks overlap.
Model A: PROMO_PROVIDER / PROMO_MODEL     Model B: PROMO_PROVIDER_B / PROMO_MODEL_B
python compare_models.py --input ./episodes --out outputs"""
import argparse
import json
from pathlib import Path

import config
from llm import LLM, LLMError
from parser import load_dir
from run_all import process, write_outputs, summary
from utils import overlap_ratio, segs_str


def by_episode(picks):
    d = {}
    for p in picks:
        d.setdefault(p["segments"][0]["episode"], []).append(p)
    return {k: sorted(v, key=lambda x: x["rank"]) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./episodes")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    pa, pb = config.provider(), config.provider("_B")
    if not pb:
        raise SystemExit("Set PROMO_PROVIDER_B and PROMO_MODEL_B for the second model.")
    episodes, _ = load_dir(a.input)
    only = set(sorted(episodes)[:a.n])
    runs = {}
    for tag, prov, mod in (("A", pa, config.model(pa)), ("B", pb, config.model(pb, "_B"))):
        out = Path(a.out) / f"compare_{tag}"
        try:
            llm = LLM(prov, mod, cache_dir=out / "cache")
        except LLMError as e:
            raise SystemExit(str(e))
        if llm.provider != "mock":
            try:
                llm._api_key()
            except LLMError as e:
                raise SystemExit(f"Model {tag}: {e}")
        print(f"Model {tag}: {llm.provider}/{llm.model}")
        res = process(episodes, llm, only=only)
        write_outputs(res, out)
        print(summary(res), "\n")
        runs[tag] = (llm, by_episode(res["picks"]))
    (la, A), (lb, B) = runs["A"], runs["B"]
    rows, t1_in_t3 = [], 0
    j_top1 = same_top1 = 0
    n_eps = 0
    for ep in sorted(only):
        if ep not in A or ep not in B:
            continue
        n_eps += 1
        a1 = A[ep][0]
        match3 = any(overlap_ratio(a1, b) >= config.HIT_OVERLAP for b in B[ep][:3])
        same = overlap_ratio(a1, B[ep][0]) >= config.HIT_OVERLAP
        t1_in_t3 += match3
        same_top1 += same
        sa = segs_str(a1).replace(" | ", " + ")
        sb = segs_str(B[ep][0]).replace(" | ", " + ")
        rows.append(f"| Ep{ep} | {sa} | {sb} | {'yes' if same else 'no'} | {'yes' if match3 else 'no'} |")
    rep = [f"# Model comparison on {n_eps} episodes",
           f"A = `{la.provider}/{la.model}`, B = `{lb.provider}/{lb.model}`. Overlap = at least {config.HIT_OVERLAP:.0%} shared screen time.\n",
           f"- Same top-1 pick: {same_top1}/{n_eps}", f"- A's top-1 appears in B's top-3: {t1_in_t3}/{n_eps}\n",
           "| Episode | A top-1 | B top-1 | same top-1 | A top-1 in B top-3 |", "|---|---|---|---|---|"] + rows
    Path(a.out, "compare_report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
