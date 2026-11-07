"""Run the whole pipeline: python run_all.py --input ./episodes --out outputs"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import blind_viewer
import candidates
import config
import failures
import post_first
import proposer
import ranker
import scorer
import validator
from llm import LLM, LLMError
from parser import load_dir
from utils import episodes_of, home_episode, segs_str


def _process_one(ep, episodes, llm, blind_k):
    n = ep.number
    cands = candidates.generate(ep)
    ai, _ = proposer.propose(llm, ep, episodes.get(n - 1), episodes.get(n + 1))
    allp = [validator.validate(p, episodes) for p in cands + ai]
    for i, p in enumerate(allp, 1):
        p["id"] = f"{p['source']}-{n}-{i}"
        p["home_episode"] = n
        p.setdefault("episodes", [n])
    ok = [p for p in allp if p["status"] == "ok"]
    scorer.score_promos(llm, n, ok, episodes)
    for p in ok:
        ranker.rank_score(p, episodes, final=False)
    ok.sort(key=lambda x: -x["rank_score"])
    tested = ok[:blind_k] if blind_k else []
    blind_viewer.run_blind(llm, n, tested, episodes)
    for p in ok:
        ranker.rank_score(p, episodes, final=True)
    pool = tested if blind_k else ok
    stats = {
        "candidates": len(cands), "ai_proposed": len(ai), "accepted": len(ok),
        "rejected": sum(p["status"] == "rejected" for p in allp),
        "repaired": sum(bool(p.get("repairs")) for p in ok),
        "unverified": sum(not p["verified"] for p in ok),
    }
    return allp, pool, stats


def process(episodes, llm, only=None, progress=None, blind_k=None, delay=None):
    blind_k = config.BLIND_TOP_K if blind_k is None else blind_k
    delay = config.delay() if delay is None else delay
    nums = [n for n in sorted(episodes) if only is None or n in only]
    res = {"all": [], "pool": [], "errors": {}, "ep_stats": {}, "aborted": ""}
    streak = 0
    for i, n in enumerate(nums):
        if streak >= 3:   # circuit breaker: a bad key / outage should not grind through every episode
            res["aborted"] = f"stopped after 3 consecutive failures; episodes {nums[i:][:5]}... not attempted"
            break
        if progress:
            progress(i, len(nums), n)
        live0 = llm.live_calls
        try:
            allp, pool, stats = _process_one(episodes[n], episodes, llm, blind_k)
        except (LLMError, ValueError, KeyError, TypeError) as e:
            res["errors"][n] = f"{type(e).__name__}: {e}"[:300]
            streak += 1
            continue
        streak = 0
        res["all"] += allp
        res["pool"] += pool
        res["ep_stats"][n] = stats
        if llm.provider != "mock" and llm.live_calls > live0 and i < len(nums) - 1:
            time.sleep(delay)
    if progress:
        progress(len(nums), len(nums), None)
    res["picks"] = ranker.pick_per_episode(res["pool"])
    res["bstats"] = blind_viewer.blind_stats([p for p in res["all"] if p["status"] == "ok"])
    res["post_first"], res["skipped"] = post_first.choose(res["picks"], [p for p in res["all"] if p["status"] == "ok"])
    res["provider"], res["model"] = llm.provider, llm.model
    res["midline_cuts"] = sum(validator.count_midline_cuts(p, episodes) for p in res["all"] if p["status"] == "ok")
    return res


CSV_COLS = ["rank", "episodes", "segments", "shape", "source", "verified", "blind_viewer_ok", "final_score",
            "hook_score", "cliffhanger_score", "standalone_score", "standalone_self_reported", "hook", "cliffhanger",
            "why", "rank_reason", "penalties", "flags"]


def csv_rows(picks):
    rows = []
    for p in picks:
        s = p["scores"]
        b = p.get("blind")
        rows.append({
            "rank": p["rank"], "episodes": ",".join(map(str, episodes_of(p))), "segments": segs_str(p),
            "shape": p["shape"], "source": p["source"], "verified": "yes" if p["verified"] else "no",
            "blind_viewer_ok": "n/a" if b is None else ("yes" if b["ok"] else "no"),
            "final_score": round(p["rank_score"], 2), "hook_score": s["hook"], "cliffhanger_score": s["cliffhanger"],
            "standalone_score": round(s.get("standalone_final", s["standalone"]), 1),
            "standalone_self_reported": s["standalone"], "hook": p["hook_line"], "cliffhanger": p["cliffhanger_line"],
            "why": p.get("why", ""), "rank_reason": p["rank_reason"],
            "penalties": ";".join(f"{k}:{v}" for k, v in p.get("penalties", {}).items()),
            "flags": ";".join(p.get("flags", [])),
        })
    return rows


def write_csv(picks, path_or_buf):
    w = csv.DictWriter(path_or_buf, fieldnames=CSV_COLS)
    w.writeheader()
    w.writerows(csv_rows(picks))


def write_outputs(res, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clean = lambda p: {k: v for k, v in p.items() if not k.startswith("_")}
    picked = {p["id"]: p["rank"] for p in res["picks"]}
    allj = []
    for p in res["all"]:
        q = clean(p)
        q["picked_rank"] = picked.get(p["id"])
        allj.append(q)
    (out / "promos_all.json").write_text(json.dumps(
        {"provider": res["provider"], "model": res["model"], "promos": allj}, indent=1, ensure_ascii=False))
    with open(out / "promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
        write_csv(res["picks"], f)
    (out / "post_first.md").write_text(post_first.to_markdown(res["post_first"], res["skipped"], res["provider"]))
    (out / "failures.md").write_text(failures.build(
        res["all"], res["picks"], res["errors"], res["ep_stats"], res["bstats"], res["provider"], res["model"]))


def summary(res):
    st = res["ep_stats"]
    c = [v["candidates"] for v in st.values()] or [0]
    ok = [p for p in res["all"] if p["status"] == "ok"]
    lines = ([f"!! ABORTED: {res['aborted']}"] if res.get("aborted") else []) + [
        f"provider/model          : {res['provider']} / {res['model']}",
        f"episodes ok / failed    : {len(st)} / {len(res['errors'])} {sorted(res['errors']) or ''}",
        f"code candidates per ep  : min {min(c)}, avg {sum(c)/len(c):.1f}, max {max(c)}",
        f"AI promos proposed      : {sum(v['ai_proposed'] for v in st.values())}",
        f"rejected by guards      : {sum(v['rejected'] for v in st.values())} "
        f"(code {sum(p['source']=='code' and p['status']=='rejected' for p in res['all'])}, "
        f"ai {sum(p['source']=='ai' and p['status']=='rejected' for p in res['all'])})",
        f"repaired by guards      : {sum(v['repaired'] for v in st.values())}",
        f"unverified quotes       : {sum(v['unverified'] for v in st.values())}",
        f"MID-LINE CUTS (must be 0): {res['midline_cuts']}  over {len(ok)} accepted promos",
        f"top picks kept          : {len(res['picks'])}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./episodes")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--episodes", help="comma list, e.g. 1,2,3 (others still used as neighbours)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--blind-k", type=int)
    ap.add_argument("--delay", type=float)
    a = ap.parse_args()
    episodes, notes = load_dir(a.input)
    for n in notes:
        print("note:", n)
    if not episodes:
        sys.exit(f"No episodes found in {a.input}")
    only = None
    if a.episodes:
        only = {int(x) for x in a.episodes.split(",")}
    elif a.limit:
        only = set(sorted(episodes)[:a.limit])
    try:
        llm = LLM(a.provider, a.model, cache_dir=Path(a.out) / "cache")
    except LLMError as e:
        sys.exit(str(e))
    if llm.provider != "mock":
        try:
            llm._api_key()
        except LLMError as e:
            sys.exit(f"{e}\nTip: use --provider mock for an offline dry run.")
    print(f"Loaded {len(episodes)} episodes; provider={llm.provider} model={llm.model}")
    res = process(episodes, llm, only=only, blind_k=a.blind_k, delay=a.delay,
                  progress=lambda i, n, e: print(f"  [{i}/{n}] episode {e}", flush=True) if e else None)
    write_outputs(res, a.out)
    print("\n" + summary(res))
    for n, msg in sorted(res["errors"].items())[:5]:
        print(f"  episode {n} failed: {msg}")
    print(f"\nWrote promos_all.json, promos_ranked.csv, post_first.md, failures.md to {a.out}/")
    print(f"API calls: {llm.live_calls} live, {llm.cached_calls} from cache")


if __name__ == "__main__":
    main()
