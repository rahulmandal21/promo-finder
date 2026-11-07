"""Code-made candidates: 12-16 safe clip proposals per episode (no AI)."""
import re

import config

SHOCK = {
    "slap", "slapped", "strike", "struck", "gasp", "scream", "screams", "collapse", "collapses",
    "blood", "dead", "death", "die", "kill", "murder", "poison", "betray", "betrayed", "betrayal",
    "lie", "lied", "liar", "secret", "truth", "never", "divorce", "married", "marry", "pregnant",
    "baby", "traitor", "duke", "prince", "king", "queen", "emperor", "arrest", "guards", "sword",
    "dagger", "run", "stop", "now", "you", "fired", "leave", "get out", "mine", "forever", "swear",
    "revenge", "destroy", "ruin", "shame", "disgrace", "whore", "liar", "impossible", "who",
}
CTX = {
    "father", "mother", "husband", "wife", "marry", "married", "sister", "brother", "duke", "lady",
    "lord", "soldier", "war", "betray", "betrayed", "engaged", "wedding", "fiance", "fiancé",
    "family", "king", "queen", "prince", "princess", "debt", "inherit", "contract", "dowry", "secret",
    "truth", "love", "hate", "child", "baby", "divorce", "wounded", "estate", "title", "bride",
}


def punch_score(text):
    w = re.findall(r"[a-z']+", text.lower())
    return sum(1 for x in w if x in SHOCK) + text.count("!") * 0.5 + text.count("?") * 0.5


def context_score(text):
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    names = sum(1 for i, w in enumerate(words) if i > 0 and w[0].isupper() and w != "I")
    kw = sum(1 for w in words if w.lower() in CTX)
    s = 2.0 * names + 1.0 * kw + min(len(words), 20) / 10.0
    return s - (1.5 if len(words) < 4 else 0.0)


def _spread(items, k):
    if len(items) <= k:
        return list(items)
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def _nearest(items, key, target, k=1):
    return sorted(items, key=lambda x: abs(key(x) - target))[:k]


def _ok_s(ep, e):
    return ep.inside_dialogue(e.start) is None


def _ok_e(ep, e):
    return ep.inside_dialogue(e.end) is None


GAP = 0.4   # joins must have a real gap, otherwise they are one continuous clip


def _seg(ep, a, b):
    return {"episode": ep.number, "start_event": a.id, "end_event": b.id}


def _span(a, b):
    return b.end - a.start


def bridge_options(ep, after_t, before_t, top=3):
    """1-3 consecutive spoken lines between two moments, 5-20 s, ranked by how much context they carry."""
    dls = [d for d in ep.dialogue if d.start >= after_t - config.TOL and d.end <= before_t + config.TOL]
    out = []
    for i in range(len(dls)):
        for n in (1, 2, 3):
            grp = dls[i:i + n]
            if len(grp) < n or not _ok_s(ep, grp[0]) or not _ok_e(ep, grp[-1]):
                continue
            dur = _span(grp[0], grp[-1])
            if config.BRIDGE_MIN <= dur <= config.BRIDGE_MAX:
                out.append({"a": grp[0], "b": grp[-1], "dur": dur,
                            "score": sum(context_score(g.text) for g in grp)})
    out.sort(key=lambda x: -x["score"])
    picked, seen = [], set()
    for o in out:
        if o["a"].id in seen:
            continue
        seen.add(o["a"].id)
        picked.append(o)
        if len(picked) == top:
            break
    return picked


def _lines(ep, segs):
    first = [e for e in ep.events_in(segs[0][0].start, segs[0][1].end) if e.type == "dialogue"]
    last = [e for e in ep.events_in(segs[-1][0].start, segs[-1][1].end) if e.type == "dialogue"]
    return (first[0].text if first else ""), (last[-1].text if last else "")


def _mk(ep, shape, segs, why):
    hook, cliff = _lines(ep, segs)
    return {
        "source": "code", "shape": shape, "shape_reason": "",
        "segments": [_seg(ep, a, b) for a, b in segs],
        "hook_line": hook, "cliffhanger_line": cliff, "why": why, "new_viewer_needs": "",
    }


def generate(ep):
    dl, evs = ep.dialogue, ep.events
    if len(dl) < 3:
        return []
    t0 = evs[0].start
    # hook start: skip a long silent opening, keep it when dialogue arrives quickly
    d0 = next((d for d in dl if _ok_s(ep, d)), dl[0])
    lead = max((e for e in evs if e.start <= d0.start + config.TOL and _ok_s(ep, e)), key=lambda e: e.start)
    starts = [evs[0]] if d0.start - t0 <= 3.0 else [lead]

    hooks = []
    for s in starts:
        ends = [d for d in dl if _ok_e(ep, d) and config.HOOK_MIN <= d.end - s.start <= config.HOOK_MAX]
        hooks += [(s, d) for d in _spread(ends, 3)]

    tails = [d for d in reversed(dl) if _ok_e(ep, d)][:2]
    twists = []
    for ee in tails:
        ss = [d for d in dl if _ok_s(ep, d) and d.start < ee.start and config.TWIST_MIN <= ee.end - d.start <= config.TWIST_MAX]
        twists += [(d, ee) for d in _spread(ss, 3)]

    def tot(*parts):
        return sum(_span(a, b) for a, b in parts)

    ht, hbt = [], []
    for h in hooks:
        for t in twists:
            if h[1].end > t[0].start - GAP:
                continue
            total = tot(h, t)
            if config.MIN_TOTAL <= total <= config.MAX_TOTAL:
                ht.append((abs(total - 38), h, t))
            for b in bridge_options(ep, h[1].end + GAP, t[0].start - GAP):
                total3 = tot(h, (b["a"], b["b"]), t)
                if config.MIN_TOTAL <= total3 <= config.MAX_TOTAL and b["dur"] < _span(*t):
                    hbt.append((b["score"] - abs(total3 - 40) * 0.05, h, (b["a"], b["b"]), t, b))
    ht.sort(key=lambda x: x[0])
    hbt.sort(key=lambda x: -x[0])

    out = []
    # hook+bridge+twist: greedy, penalise reusing the same bridge so the set varies
    used = {}
    for _ in range(6):
        best = max(((sc - 1.5 * used.get(b[0].id, 0), i) for i, (sc, h, b, t, _o) in enumerate(hbt)
                    if not any(x["segments"] == [_seg(ep, *h), _seg(ep, *b), _seg(ep, *t)] for x in out)),
                   default=None)
        if not best:
            break
        sc, h, b, t, _o = hbt[best[1]]
        used[b[0].id] = used.get(b[0].id, 0) + 1
        out.append(_mk(ep, "hook+bridge+twist", [h, b, t],
                       "Code-built: opening hook, the most context-rich lines as bridge, closing twist."))
    for _s, h, t in ht[:3]:
        out.append(_mk(ep, "hook+twist", [h, t], "Code-built: opening hook joined to the closing twist."))

    ends_ = []
    for ee in tails:
        ss = [d for d in dl if _ok_s(ep, d) and config.MIN_TOTAL <= ee.end - d.start <= config.MAX_TOTAL]
        for tgt in ((30, 45) if ee is tails[0] else (35,)):
            ends_ += [(d, ee) for d in _nearest(ss, lambda x: ee.end - x.start, tgt)]
    for a, b in ends_:
        out.append(_mk(ep, "single", [(a, b)], "Code-built: closing stretch ending on the last spoken line."))
    ends = [d for d in dl if _ok_e(ep, d) and config.MIN_TOTAL <= d.end - starts[-1].start <= config.MAX_TOTAL]
    for tgt in (30, 50):
        for d in _nearest(ends, lambda x: x.end - starts[-1].start, tgt):
            out.append(_mk(ep, "single", [(starts[-1], d)], "Code-built: opening stretch from the first spoken line."))

    seen, uniq = set(), []
    for p in out:
        k = tuple((s["start_event"], s["end_event"]) for s in p["segments"])
        if k not in seen and p["hook_line"] and p["cliffhanger_line"]:
            seen.add(k)
            uniq.append(p)
    return uniq[:config.CANDS_PER_EPISODE[1]]
