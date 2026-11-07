"""AI step 3: a viewer who sees ONLY the clip text - no logline, no notes, no other episodes."""
import candidates as C
from scorer import clip_lines
import config

SYSTEM = """You are a viewer scrolling social media. You have never seen this show and know nothing about it. \
Below is everything you get: a short clip, written out. Judge only what is on the page.
Reply with ONE strict JSON object only:
{"who":"<who are these people and how are they related?>","happening":"<what is happening?>",\
"want_to_know":"<what do you want to know next?>","makes_sense":"yes or no - does this make sense on its own?",\
"confusion":"<what confused you, or empty>"}"""


def _ask(llm, lines, episode):
    user = "THE CLIP:\n" + "\n".join(lines)
    obj, usage, cached = llm.call_json("blind", SYSTEM, user, meta={"lines": lines}, episode=episode)
    if not isinstance(obj, dict):
        raise ValueError("blind viewer returned non-object")
    return {
        "ok": str(obj.get("makes_sense", "")).strip().lower().startswith("y"),
        "who": str(obj.get("who", "")), "happening": str(obj.get("happening", "")),
        "want_to_know": str(obj.get("want_to_know", "")), "confusion": str(obj.get("confusion", "")),
    }


def _with_bridge_lines(p, episodes):
    """Same clip, plus the best code-chosen bridge between its two segments (hook+twist clips only)."""
    if len(p["segments"]) != 2 or p["segments"][0]["episode"] != p["segments"][1]["episode"]:
        return None
    a, b = p["segments"]
    ep = episodes[a["episode"]]
    opts = C.bridge_options(ep, a["end"] + C.GAP, b["start"] - C.GAP, top=1)
    if not opts:
        return None
    br = {"segments": [a, {"episode": a["episode"], "start": opts[0]["a"].start, "end": opts[0]["b"].end}, b]}
    return clip_lines(br, episodes, with_time=False)


def run_blind(llm, ep_number, promos, episodes):
    """Blind-test the given promos one clip per call. Mutates promos['blind'] and the final standalone."""
    for p in promos:
        lines = clip_lines(p, episodes, with_time=False)
        res = _ask(llm, lines, ep_number)
        res["bridge_ok"] = res["bridge_fixed"] = None
        bl = _with_bridge_lines(p, episodes)
        if bl is not None:
            res["bridge_ok"] = _ask(llm, bl, ep_number)["ok"]
            res["bridge_fixed"] = (not res["ok"]) and res["bridge_ok"]
        p["blind"] = res
        s = p["scores"]["standalone"]
        p["scores"]["standalone_final"] = s if res["ok"] else max(0.0, s - config.BLIND_NO_PENALTY)


def blind_stats(promos):
    tested = [p for p in promos if p.get("blind")]
    no = [p for p in tested if not p["blind"]["ok"]]
    ht = [p for p in tested if p["blind"]["bridge_ok"] is not None]
    ht_no = [p for p in ht if not p["blind"]["ok"]]
    fixed = [p for p in ht_no if p["blind"]["bridge_ok"]]
    dis = [p for p in tested if (p["scores"]["standalone"] >= 6) != p["blind"]["ok"]]
    return {
        "tested": len(tested), "blind_no": len(no),
        "hook_twist_with_bridge_tested": len(ht), "hook_twist_blind_no": len(ht_no),
        "bridge_turned_no_into_yes": len(fixed),
        "bridge_fix_rate": (len(fixed) / len(ht_no)) if ht_no else None,
        "self_vs_blind_disagree": len(dis),
        "self_vs_blind_disagree_rate": (len(dis) / len(tested)) if tested else None,
    }
