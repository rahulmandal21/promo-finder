"""Score = 0.35*hook + 0.35*cliffhanger + 0.30*standalone (config.py), minus penalties.
Keep the top 1-3 per episode with no overlaps or near-duplicates, then rank everything together."""
import re
from collections import defaultdict

import candidates as C
import config
from utils import home_episode, norm, overlap_ratio


def base_score(p, final=True):
    s = p["scores"]
    stand = s.get("standalone_final", s["standalone"]) if final else s["standalone"]
    w = config.WEIGHTS
    return w["hook"] * s["hook"] + w["cliffhanger"] * s["cliffhanger"] + w["standalone"] * stand


def quiet_start(p, episodes):
    seg = p["segments"][0]
    ep = episodes[seg["episode"]]
    early = ep.events_in(seg["start"], seg["start"] + config.HOOK_WINDOW)
    spoken = [e for e in early if e.type == "dialogue"]
    shock_action = any(e.type == "action" and C.punch_score(e.text) >= 1.5 for e in early)
    if not spoken:
        return not shock_action
    greet = re.match(r"^(good (morning|evening|afternoon)|hello|hi\b|hey\b|welcome|please\b|thank)",
                     spoken[0].text.strip().lower())
    return bool(greet)


def penalties(p, episodes, use_blind=True):
    flags = " ".join(p.get("flags", []))
    pen = {}
    if not p.get("verified"):
        pen["unverified"] = config.PENALTIES["unverified"]
    if quiet_start(p, episodes):
        pen["quiet_start"] = config.PENALTIES["quiet_start"]
    cue_end = any(e.is_cue for e in episodes[p["segments"][-1]["episode"]].events_in(
        p["segments"][-1]["start"], p["segments"][-1]["end"]))
    if "cliffhanger_cue_only" in flags or (cue_end and p["scores"]["cliffhanger"] < 6):
        pen["cue_only"] = config.PENALTIES["cue_only"]
    if use_blind and (p.get("blind") and not p["blind"]["ok"]):
        pen["needs_earlier"] = config.PENALTIES["needs_earlier"]
    if "hook_late" in flags:
        pen["hook_late"] = config.PENALTIES["hook_late"]
    return pen


def rank_score(p, episodes, final=True):
    pen = penalties(p, episodes, use_blind=final)
    p["penalties"] = pen
    p["rank_score"] = round(base_score(p, final) - sum(pen.values()), 3)
    return p["rank_score"]


def needs_earlier(p):
    b = p.get("blind")
    if b is not None:
        return not b["ok"]
    return p["scores"]["standalone"] <= 3


def near_duplicate(p, q):
    if overlap_ratio(p, q) >= config.DUP_OVERLAP:
        return True
    return bool(norm(p.get("hook_line")) and norm(p.get("hook_line")) == norm(q.get("hook_line")))


def pick_per_episode(promos):
    """promos: scored+ranked dicts. -> list of picks (top 1-3 per home episode)."""
    by_ep = defaultdict(list)
    for p in promos:
        by_ep[home_episode(p)].append(p)
    picks = []
    for ep, lst in by_ep.items():
        lst.sort(key=lambda x: -x["rank_score"])
        chosen = []
        for p in lst:
            if len(chosen) >= config.KEEP_PER_EPISODE:
                break
            if chosen and p["rank_score"] < config.MIN_EXTRA_SCORE:
                break
            if any(near_duplicate(p, q) for q in chosen):
                continue
            chosen.append(p)
        picks += chosen
    picks.sort(key=lambda x: -x["rank_score"])
    for i, p in enumerate(picks, 1):
        p["rank"] = i
        s = p["scores"]
        top = max(("hook", "cliffhanger", "standalone"), key=lambda k: s[k])
        p["rank_reason"] = (s.get("reason") or p.get("why") or "").strip() or f"strongest on {top}"
    return picks


def shape_share(picks, n=10):
    top = picks[:n] if n else picks
    if not top:
        return None, 0.0
    from collections import Counter
    shape, c = Counter(p["shape"] for p in top).most_common(1)[0]
    return shape, c / len(top)
