"""Builds failures.md automatically from a run. Nothing here is hand-written per run."""
from collections import Counter

import config
from ranker import needs_earlier, shape_share
from utils import home_episode, segs_str


def _row(p):
    return f"- {segs_str(p)} [{p['source']}, {p['shape']}]"


def build(all_promos, picks, errors, ep_stats, bstats, provider, model):
    ok = [p for p in all_promos if p["status"] == "ok"]
    rej = [p for p in all_promos if p["status"] == "rejected"]
    L = ["# failures.md (auto-generated)\n"]
    if provider == "mock":
        L.append("> **MOCK RUN.** Scores/blind answers are heuristics from a test provider, so score-based sections "
                 "(low-scoring, blind-test, monoculture) describe the mock, not a real model. "
                 "Guard sections (rejected / repaired / unverified) are real behaviour of the validator.\n")
    L.append(f"Provider/model: `{provider}` / `{model}`. Episodes processed: {len(ep_stats)}; "
             f"episodes failed: {len(errors)}; promos considered: {len(all_promos)}; accepted: {len(ok)}; "
             f"rejected: {len(rej)}.\n")
    if errors:
        L.append("## Episodes that failed to run (re-run to retry; successful calls are cached)\n")
        L += [f"- Episode {k}: {v}" for k, v in sorted(errors.items())]
    L.append("\n## Rejected by the guards\n")
    rc = Counter(p["reject_reason"].split(" (")[0] for p in rej)
    L.append(", ".join(f"{k}: {v}" for k, v in rc.most_common()) or "none")
    by_src = Counter(p["source"] for p in rej)
    L.append(f"\nBy source: {dict(by_src)}")
    L += [_row(p) + f" - {p['reject_reason']}" for p in rej[:25]]
    rep = [p for p in ok if p.get("repairs")]
    L.append(f"\n## Repaired by the guards ({len(rep)})\n")
    L += [_row(p) + f" - {', '.join(sorted(set(p['repairs'])))}" for p in rep[:25]]
    unv = [p for p in ok if not p["verified"]]
    L.append(f"\n## Unverified quotes - hook or cliffhanger line not found where claimed ({len(unv)})\n")
    L += [_row(p) + f" - {'; '.join(f for f in p['flags'] if 'not found' in f)}" for p in unv[:25]]
    scored = [p for p in picks if p.get("scores")]
    low = [p for p in scored if p["rank_score"] < 6]
    L.append(f"\n## Top picks that still score low (< 6.0) ({len(low)})\n")
    L += [_row(p) + f" - score {p['rank_score']:.1f}" for p in low[:25]]
    bf = [p for p in all_promos if p.get("blind") and not p["blind"]["ok"]]
    L.append(f"\n## Failed the blind-viewer test ({len(bf)} of {bstats['tested']} tested)\n")
    L += [_row(p) + f" - {p['blind']['confusion'] or 'confused'}" for p in bf[:25]]
    eps_all = sorted(ep_stats)
    best = {}
    for p in ok:
        if p.get("scores"):
            e = home_episode(p)
            best[e] = max(best.get(e, 0), p["scores"]["cliffhanger"])
    nocliff = [e for e in eps_all if e in best and best[e] < 6]
    L.append(f"\n## Episodes with no real cliffhanger (best cliffhanger score < 6) ({len(nocliff)})\n")
    L.append(", ".join(f"Ep{e} ({best[e]:.1f})" for e in nocliff) or "none")
    top1 = {}
    for p in picks:
        top1.setdefault(home_episode(p), p)
    dep = [e for e, p in sorted(top1.items()) if needs_earlier(p)]
    L.append(f"\n## Episodes whose best pick only makes sense after earlier episodes ({len(dep)})\n")
    L.append(", ".join(f"Ep{e}" for e in dep) or "none")
    shape, share = shape_share(picks, 0)
    L.append("\n## Do the promos all look alike?\n")
    L.append(f"Most common shape among all {len(picks)} top picks: **{shape}** at {share:.0%}. " +
             ("**FLAG: more than 70% share one shape - the promos would all look alike.**" if share > config.MONOCULTURE
              else "Below the 70% flag threshold."))
    L.append(f"Shape counts: {dict(Counter(p['shape'] for p in picks))}")
    L.append("\n## Self-reported standalone vs blind viewer\n")
    L.append(f"Disagreements: {bstats['self_vs_blind_disagree']} of {bstats['tested']} "
             f"({(bstats['self_vs_blind_disagree_rate'] or 0):.0%}). Rule: self score >= 6 counts as 'yes'.")
    fr = bstats["bridge_fix_rate"]
    L.append(f"Hook+twist clips tested with and without a bridge: {bstats['hook_twist_with_bridge_tested']}; "
             f"blind said no: {bstats['hook_twist_blind_no']}; bridge turned no into yes: "
             f"{bstats['bridge_turned_no_into_yes']}" + (f" ({fr:.0%})." if fr is not None else " (no 'no' cases to fix)."))
    src = Counter(p["source"] for p in picks)
    L.append(f"\n## Who supplied the winners?\nTop picks by source: {dict(src)}")
    return "\n".join(L) + "\n"
