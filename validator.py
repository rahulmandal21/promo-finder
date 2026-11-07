"""Code guards. Never trust the model: snap, repair or reject, then verify quoted lines."""
import re

import config
from utils import norm, promo_duration, seg_dur

ID_RE = re.compile(r"^E(\d+)-(\d+)$")
SHAPES = ("single", "hook+twist", "hook+bridge+twist", "other")


def _event(eid, episodes):
    m = ID_RE.match(str(eid).strip().upper())
    if not m:
        raise ValueError(f"bad event id '{eid}'")
    ep = episodes.get(int(m.group(1)))
    if not ep or eid_key(eid) not in ep.by_id:
        raise ValueError(f"unknown event id '{eid}'")
    return ep, ep.by_id[eid_key(eid)]


def eid_key(eid):
    return str(eid).strip().upper()


def _resolve(seg, episodes):
    """-> list of resolved segments (a segment spanning two episodes is split in two)."""
    ep_a, a = _event(seg.get("start_event"), episodes)
    ep_b, b = _event(seg.get("end_event"), episodes)
    if ep_a.number == ep_b.number:
        if b.end <= a.start:
            raise ValueError("segment ends before it starts")
        return [dict(episode=ep_a.number, start=a.start, end=b.end, start_event=a.id, end_event=b.id)], False
    if ep_b.number < ep_a.number:
        raise ValueError("segment runs backwards across episodes")
    if ep_b.number - ep_a.number > 1:
        raise ValueError("segment spans more than two episodes")
    last = ep_a.events[-1]
    first = ep_b.events[0]
    return [
        dict(episode=ep_a.number, start=a.start, end=max(e.end for e in ep_a.events), start_event=a.id, end_event=last.id),
        dict(episode=ep_b.number, start=first.start, end=b.end, start_event=first.id, end_event=b.id),
    ], True


def _find_line(quote, events):
    """Return the event whose text the quote really comes from, else None.
    A partial quote of a real line is fine; invented extra words are not."""
    q = norm(quote)
    if not q:
        return None
    qt = q.split()
    for e in events:
        t = norm(e.text)
        tt = t.split()
        if q == t or (len(qt) >= 2 and q in t):
            return e
        if len(qt) >= 2 and t in q and len(qt) <= 1.15 * len(tt):   # same line, trivial wording noise
            return e
        if len(qt) >= 3:
            tset = set(tt)
            missing = sum(1 for w in qt if w not in tset)
            if missing <= int(0.1 * len(qt)):
                return e
    return None


def validate(promo, episodes):
    p = dict(promo)
    p.setdefault("flags", [])
    p["flags"] = list(p["flags"])
    p["repairs"] = []
    p["status"], p["reject_reason"] = "ok", ""
    p["verified"] = p["hook_verified"] = p["cliffhanger_verified"] = False
    p["repaired"] = False

    def reject(why):
        p["status"], p["reject_reason"] = "rejected", why
        return p

    raw = p.get("segments") or []
    if not raw:
        return reject("no segments")
    if len(raw) > config.MAX_SEGMENTS:
        return reject(f"more than {config.MAX_SEGMENTS} segments")

    # 1. resolve ids -> times (split cross-episode segments)
    segs = []
    try:
        for s in raw:
            parts, split = _resolve(s, episodes)
            if split:
                p["repairs"].append("split cross-episode segment")
            segs += parts
    except ValueError as e:
        return reject(str(e))

    # 2. snap every boundary out of spoken lines
    for s in segs:
        ep = episodes[s["episode"]]
        t, d = ep.snap_start(s["start"])
        if d is not None:
            s["start"], s["start_event"] = t, d.id
            p["repairs"].append("start moved to spoken-line start")
        t, d = ep.snap_end(s["end"])
        if d is not None:
            s["end"], s["end_event"] = t, d.id
            p["repairs"].append("end extended to spoken-line end")
        if s["end"] - s["start"] <= 0:
            return reject("empty segment after snapping")

    # 3. story order, merge touching/overlapping segments
    merged = [segs[0]]
    for s in segs[1:]:
        m = merged[-1]
        if (s["episode"], s["start"]) < (m["episode"], m["start"]):
            return reject("segments not in story order")
        if s["episode"] == m["episode"] and s["start"] <= m["end"] + 0.25:
            if s["end"] > m["end"]:
                m["end"], m["end_event"] = s["end"], s["end_event"]
            p["repairs"].append("merged touching segments")
        else:
            merged.append(s)
    p["segments"] = merged
    p["episodes"] = sorted({s["episode"] for s in merged}, key=lambda x: [s["episode"] for s in merged].index(x))

    # 4. total length
    dur = promo_duration(p)
    p["duration"] = round(dur, 2)
    if dur < config.MIN_TOTAL - config.TOL:
        return reject(f"too short ({dur:.1f}s)")
    if dur > config.MAX_TOTAL + config.TOL:
        return reject(f"too long ({dur:.1f}s)")

    # 5. shape label must match the segments
    n = len(merged)
    expected = {1: "single", 2: "hook+twist", 3: "hook+bridge+twist"}.get(n, "other")
    claimed = str(p.get("shape") or "").strip().lower()
    if claimed == "other" and p.get("shape_reason"):
        p["shape"] = "other"
    elif claimed == expected:
        p["shape"] = expected
    else:
        if claimed:
            p["flags"].append(f"shape_relabelled ({claimed or 'none'} -> {expected})")
        p["shape"] = expected
    if p["shape"] == "other" and not p.get("shape_reason"):
        p["flags"].append("other_shape_without_reason")
    if p["shape"] == "hook+bridge+twist" and seg_dur(merged[1]) >= seg_dur(merged[2]):
        return reject("bridge is not shorter than the twist")

    # 6. verify quoted hook and cliffhanger really sit in the script where claimed
    first_ep = episodes[merged[0]["episode"]]
    last_ep = episodes[merged[-1]["episode"]]
    fe = first_ep.events_in(merged[0]["start"], merged[0]["end"])
    le = last_ep.events_in(merged[-1]["start"], merged[-1]["end"])
    hook_ev = _find_line(p.get("hook_line", ""), [e for e in fe if e.type in ("dialogue", "action")])
    cliff_ev = _find_line(p.get("cliffhanger_line", ""), [e for e in le if e.type in ("dialogue", "action", "sound")])
    p["hook_verified"], p["cliffhanger_verified"] = hook_ev is not None, cliff_ev is not None
    if not hook_ev:
        p["flags"].append("hook_line not found in first segment")
    elif hook_ev.start - merged[0]["start"] > 3.5:
        p["flags"].append("hook_late")
    if not cliff_ev:
        p["flags"].append("cliffhanger_line not found in last segment")
    elif cliff_ev.type == "sound":
        p["flags"].append("cliffhanger_cue_only")
    p["verified"] = bool(hook_ev and cliff_ev)
    if p["repairs"]:
        p["repaired"] = True
    return p


def count_midline_cuts(p, episodes):
    """Independent re-check on final times (does not reuse the snapping code path)."""
    bad = 0
    for s in p["segments"]:
        ep = episodes[s["episode"]]
        for d in ep.dialogue:
            if d.start + config.TOL < s["start"] < d.end - config.TOL:
                bad += 1
            if d.start + config.TOL < s["end"] < d.end - config.TOL:
                bad += 1
    return bad
