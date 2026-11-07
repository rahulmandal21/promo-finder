import re


def fmt_t(x):
    m = int(x // 60)
    return f"{m:02d}:{x - 60 * m:04.1f}"


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", (s or "").lower())).strip()


def seg_dur(s):
    return s["end"] - s["start"]


def promo_duration(p):
    return sum(seg_dur(s) for s in p["segments"])


def segs_str(p):
    return " | ".join(
        f"Ep{s['episode']} {fmt_t(s['start'])}-{fmt_t(s['end'])}" if "start" in s
        else f"{s.get('start_event')}..{s.get('end_event')}" for s in p["segments"])


def home_episode(p):
    return p["segments"][0]["episode"]


def episodes_of(p):
    out = []
    for s in p["segments"]:
        if s["episode"] not in out:
            out.append(s["episode"])
    return out


def _merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _by_ep(p):
    d = {}
    for s in p["segments"]:
        d.setdefault(s["episode"], []).append((s["start"], s["end"]))
    return {k: _merge(v) for k, v in d.items()}


def overlap_ratio(p, q):
    """Seconds of shared screen time / the shorter clip's length. Works for promos or gold dicts."""
    a, b = _by_ep(p), _by_ep(q)
    inter = 0.0
    for ep in set(a) & set(b):
        for x0, x1 in a[ep]:
            for y0, y1 in b[ep]:
                inter += max(0.0, min(x1, y1) - max(x0, y0))
    da = sum(x1 - x0 for v in a.values() for x0, x1 in v)
    db = sum(x1 - x0 for v in b.values() for x0, x1 in v)
    m = min(da, db)
    return inter / m if m > 0 else 0.0
