"""AI step 2: re-score EVERY promo (code-made or AI-proposed) in one judging call, blind to who proposed it."""
import hashlib
import random

from utils import fmt_t, promo_duration

SYSTEM = """You are a strict judge of short-drama promos for Instagram/TikTok/YouTube Shorts. \
You will see several candidate clips from one episode. You do not know who chose them; judge only what is on the page.
Score each clip 0-10 (use the whole range; most clips are mediocre):
- hook: would the first 3 seconds stop a scrolling viewer? (slap, reveal, shocking line = high; quiet setup = low)
- cliffhanger: does the clip end on an unresolved moment that makes the viewer need the next part? \
(a "next beat" line shows what comes right after the clip - if the clip already answers the question, score low)
- standalone: could someone who has never seen the show follow who these people are and what is at stake?
- bridge_context: ONLY for clips made of 3 joined segments - do the middle lines give enough context for the \
last segment to be understood? 0-10. Otherwise null.
Reply with ONE strict JSON object only:
{"scores":[{"id":"P1","hook":0,"cliffhanger":0,"standalone":0,"bridge_context":null,"reason":"<one short line>"}]}"""


def clip_lines(promo, episodes, kinds=("dialogue", "action"), with_time=True):
    out, clock = [], 0.0
    for i, s in enumerate(promo["segments"]):
        if i:
            out.append("--- CUT ---")
        ep = episodes[s["episode"]]
        for e in ep.events_in(s["start"], s["end"]):
            if e.type not in kinds:
                continue
            t = f"[{clock + e.start - s['start']:4.1f}s] " if with_time else ""
            if e.type == "dialogue":
                out.append(f'{t}{e.speaker} ({e.tone}): "{e.text}"')
            else:
                out.append(f"{t}(action) {e.text}")
        clock += s["end"] - s["start"]
    return out


def next_beat(promo, episodes):
    s = promo["segments"][-1]
    ep = episodes[s["episode"]]
    nxt = [d for d in ep.dialogue if d.start >= s["end"] - 0.05]
    return f'{nxt[0].speaker}: "{nxt[0].text}"' if nxt else "(the episode ends here)"


def _clip(v):
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def score_promos(llm, ep_number, promos, episodes):
    """Mutates promos: adds p['scores']. Returns number of promos the model failed to score."""
    if not promos:
        return 0
    order = list(range(len(promos)))
    random.Random(int(hashlib.sha256(str(ep_number).encode()).hexdigest(), 16)).shuffle(order)
    blocks, clips_meta = [], []
    for label_i, idx in enumerate(order, 1):
        p = promos[idx]
        lines = clip_lines(p, episodes)
        nb = next_beat(p, episodes)
        pid = f"P{label_i}"
        p["_judge_id"] = pid
        blocks.append(f"{pid} | {len(p['segments'])} segment(s) | {promo_duration(p):.0f} s\n" + "\n".join(lines)
                      + f"\nNEXT BEAT (not in the clip): {nb}")
        clips_meta.append({"id": pid, "lines": lines, "shape": p["shape"], "next_beat": nb,
                           "n_segments": len(p["segments"]), "duration": promo_duration(p)})
    user = "Score these clips.\n\n" + "\n\n".join(blocks)
    obj, usage, cached = llm.call_json("score", SYSTEM, user, meta={"clips": clips_meta}, episode=ep_number)
    got = {str(r.get("id")): r for r in (obj.get("scores", []) if isinstance(obj, dict) else []) if isinstance(r, dict)}
    missing = 0
    for p in promos:
        r = got.get(p.pop("_judge_id"))
        if not r:
            missing += 1
            p["scores"] = {"hook": 0.0, "cliffhanger": 0.0, "standalone": 0.0, "bridge_context": None,
                           "reason": "not scored by judge"}
            p["flags"].append("score_missing")
            continue
        bc = r.get("bridge_context")
        p["scores"] = {"hook": _clip(r.get("hook")), "cliffhanger": _clip(r.get("cliffhanger")),
                       "standalone": _clip(r.get("standalone")),
                       "bridge_context": None if bc is None or len(p["segments"]) != 3 else _clip(bc),
                       "reason": str(r.get("reason") or "")}
    return missing
