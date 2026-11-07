"""Offline 'model' for testing the pipeline without an API key. It is a set of heuristics, NOT an AI:
its scores say nothing about real quality. It deliberately injects the mistakes real models make
(invented quotes, mid-line cuts, over-long clips, out-of-order segments) so the guards get exercised."""
import hashlib
import json
import random
import re

SHOCK = {"slap", "dead", "die", "kill", "betray", "betrayed", "lie", "liar", "secret", "truth", "never", "divorce",
         "married", "marry", "pregnant", "baby", "traitor", "duke", "arrest", "guards", "sword", "revenge", "ruin",
         "shame", "impossible", "who", "stop", "leave", "mine", "forever", "swear", "poison", "blood", "prince", "king"}


def _shock(t):
    w = re.findall(r"[a-z']+", t.lower())
    return sum(1 for x in w if x in SHOCK) + t.count("!") * 0.5 + t.count("?") * 0.5


def _rng(*parts):
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16))


def _tok(s):
    return max(1, len(s) // 4)


def mock_call(task, system, user, meta):
    out = {"propose": _propose, "score": _score, "blind": _blind}[task](meta, user)
    text = json.dumps(out)
    return text, {"in": _tok(system) + _tok(user), "out": _tok(text)}


def _propose(meta, user):
    ev = meta["events"]
    rnd = _rng("p", meta["episode"], len(ev))
    dl = [e for e in ev if e["type"] == "dialogue"]
    if len(dl) < 4:
        return {"promos": []}
    first, last = dl[0], dl[-1]

    def seg(a, b):
        return {"start_event": a["id"], "end_event": b["id"]}

    hook_end = [d for d in dl if d["end"] - first["start"] <= 10][-1]
    tw = [d for d in dl if last["end"] - d["start"] <= 18 and d["start"] > hook_end["end"] + 1]
    twist = (tw[0] if tw else dl[-2], last)
    mid = [d for d in dl if hook_end["end"] + 1 < d["start"] and d["end"] < twist[0]["start"] - 1]
    promos = []
    if mid:
        br = max(mid, key=lambda d: len(d["text"].split()) + 2 * sum(w[:1].isupper() for w in d["text"].split()[1:]))
        promos.append({"segments": [seg(first, hook_end), seg(br, br), seg(*twist)], "shape": "hook+bridge+twist",
                       "hook_line": first["text"], "cliffhanger_line": last["text"],
                       "why": "Opens on the first spoken question, one context line, then the closing twist.",
                       "new_viewer_needs": "Who the two main characters are."})
    promos.append({"segments": [seg(first, hook_end), seg(*twist)], "shape": "hook+twist",
                   "hook_line": first["text"], "cliffhanger_line": last["text"],
                   "why": "Straight hook-to-twist cut.", "new_viewer_needs": ""})
    pk = max(dl[1:-1], key=lambda d: _shock(d["text"]))
    after = [d for d in dl if d["start"] >= pk["start"] and d["end"] - pk["start"] <= 30]
    promos.append({"segments": [seg(pk, after[-1])], "shape": "single",
                   "hook_line": pk["text"], "cliffhanger_line": after[-1]["text"],
                   "why": "Starts on the most shocking line in the episode and skips setup.", "new_viewer_needs": ""})
    # ---- injected mistakes (deterministic) ----
    if rnd.random() < 0.35:   # invented quote
        bad = json.loads(json.dumps(promos[0]))
        bad["hook_line"] = "I never wanted any of this, " + first["text"].lower()
        bad["why"] = "(mock: hallucinated quote)"
        promos.append(bad)
    if rnd.random() < 0.25:   # too long
        promos.append({"segments": [seg(ev[0], ev[-1])], "shape": "single", "hook_line": first["text"],
                       "cliffhanger_line": last["text"], "why": "(mock: whole episode)", "new_viewer_needs": ""})
    if rnd.random() < 0.25:   # out of order
        promos.append({"segments": [seg(*twist), seg(first, hook_end)], "shape": "hook+twist",
                       "hook_line": first["text"], "cliffhanger_line": last["text"],
                       "why": "(mock: twist before hook)", "new_viewer_needs": ""})
    cams = [e for e in ev if e["type"] != "dialogue"
            and any(d["start"] + 0.3 < e["start"] < d["end"] - 0.3 for d in dl)]
    if cams and rnd.random() < 0.6:   # starts inside a spoken line
        c = rnd.choice(cams)
        d = next(d for d in dl if d["start"] + 0.3 < c["start"] < d["end"] - 0.3)
        aft = [x for x in dl if x["start"] >= d["start"] and x["end"] - d["start"] <= 28]
        promos.append({"segments": [seg(c, aft[-1])], "shape": "single", "hook_line": d["text"],
                       "cliffhanger_line": aft[-1]["text"], "why": "(mock: begins inside a spoken line)",
                       "new_viewer_needs": ""})
    return {"promos": promos}


def _score(meta, user):
    res = []
    for c in meta["clips"]:
        r = _rng("s", c["id"], "|".join(c["lines"])[:200])
        spoken = [l for l in c["lines"] if '"' in l and "(action)" not in l]
        first3 = " ".join(l for l in c["lines"][:3])
        last = spoken[-1] if spoken else ""
        lw = len(re.findall(r"[A-Za-z']+", last))
        speakers = {re.sub(r"^\[.*?\]\s*", "", l).split(" (")[0] for l in spoken}
        words = sum(len(l.split()) for l in c["lines"])
        hook = 3 + 1.5 * min(_shock(first3), 3) + r.uniform(-1, 1)
        cliff = 3 + (2.5 if re.search(r'[?!]"?$|\.\.\."?$', last.strip()) else 0) + (1.5 if lw <= 6 else 0) \
            + min(_shock(last), 2) + r.uniform(-1, 1)
        stand = 2.5 + min(len(speakers), 3) + (1.5 if c["n_segments"] == 3 else 0) + (1 if words > 70 else 0) + r.uniform(-1, 1)
        cl = lambda x: round(max(0, min(10, x)), 1)
        res.append({"id": c["id"], "hook": cl(hook), "cliffhanger": cl(cliff), "standalone": cl(stand),
                    "bridge_context": cl(4 + min(words, 100) / 25 + r.uniform(-1, 1)) if c["n_segments"] == 3 else None,
                    "reason": "(mock heuristic: shock words early, open question at the end, speakers present)"})
    return {"scores": res}


def _blind(meta, user):
    lines = meta["lines"]
    r = _rng("b", "|".join(lines)[:300])
    spoken = [l for l in lines if '"' in l and "(action)" not in l]
    speakers = {l.split(" (")[0] for l in spoken}
    words = sum(len(l.split()) for l in lines)
    ok = len(speakers) >= 2 and words >= 55 and r.random() < 0.9
    return {"who": "Several characters in a tense family or court confrontation (mock).",
            "happening": "A dispute reaches a turning point (mock).", "want_to_know": "What happens next (mock).",
            "makes_sense": "yes" if ok else "no",
            "confusion": "" if ok else "Unclear who the people are and why this matters (mock)."}
