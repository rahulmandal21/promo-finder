"""AI step 1: act as a short-drama promo editor and propose clips (creative)."""
from utils import fmt_t

SYSTEM = """You are a senior promo editor for vertical short dramas (ReelShort / DramaBox style). \
You pick clips from screenplays that make a cold scroller on Instagram, TikTok or YouTube Shorts stop, \
watch, and want the next episode.

A promo is 20-60 seconds in total. It is one continuous stretch, or a few moments joined in story order.
It must:
1. open on a HOOK inside its first 3 seconds (a slap, a reveal, a shocking line);
2. end on an unresolved CLIFFHANGER - cut one beat BEFORE the answer;
3. make sense to someone who has never seen the show.

PREFERRED SHAPE - hook+bridge+twist, three segments in story order:
 HOOK (3-12 s): the opening suspense moment that raises a question.
 BRIDGE (5-20 s, shorter than the twist): the minimum backstory so a new viewer knows who these people are \
and what is at stake - the 1-3 lines carrying the most context, nothing else.
 TWIST (10-25 s): the main climax, cut one beat before the answer.
You may use any other shape when it makes a stronger promo: a mid-episode moment, starting on the most \
shocking line and skipping setup, or ending a line or two before a reveal. If you do not use the preferred \
shape, set "shape" to "other" (or "single" / "hook+twist") and say why in "shape_reason". \
Propose a cross-episode promo only if the neighbouring-episode lines provided make it clear to a new viewer.

Rules:
- Never start or end inside a spoken line. Each segment runs from the START of start_event to the END of end_event.
- Use event IDs exactly as written in the script (e.g. E3-017). Segments must be in story order.
- hook_line and cliffhanger_line must be copied VERBATIM from a dialogue or action line inside your segments; \
hook_line from the first segment, cliffhanger_line from the last.
- SOUND FX lines that say "cliffhanger" are the writers' notes, not evidence. Judge the story itself.
- Reply with ONE strict JSON object only. No prose, no markdown."""

FORMAT = """Return JSON exactly like:
{"promos":[{"segments":[{"start_event":"E3-004","end_event":"E3-009"},{"start_event":"E3-031","end_event":"E3-036"}],
"shape":"hook+twist","shape_reason":"","hook_line":"<verbatim>","cliffhanger_line":"<verbatim>",
"why":"<one line>","new_viewer_needs":"<what a new viewer must know to follow it>"}]}
Propose 4 to 6 promos, best first."""


def _line(e, limit):
    t = e.text if len(e.text) <= limit else e.text[:limit - 1] + "…"
    who = f"{e.speaker} ({e.tone})" if e.type == "dialogue" else e.type.upper()
    return f'{e.id} | {fmt_t(e.start)}-{fmt_t(e.end)} | {who}: {t}'


def render_events(events):
    lim = {"dialogue": 400, "action": 200, "camera": 120, "sound": 100}
    return "\n".join(_line(e, lim[e.type]) for e in events)


def build_prompt(ep, prev=None, nxt=None):
    parts = [f"EPISODE {ep.number}", f"LOGLINE: {ep.logline}", f"CONTINUITY NOTES: {ep.continuity}", "", "SCRIPT:",
             render_events(ep.events)]
    if prev:
        parts += ["", f"END OF PREVIOUS EPISODE {prev.number} (for cross-episode promos only):",
                  render_events(prev.events[-10:])]
    if nxt:
        parts += ["", f"START OF NEXT EPISODE {nxt.number} (for cross-episode promos only):",
                  render_events(nxt.events[:10])]
    parts += ["", FORMAT]
    return "\n".join(parts)


def propose(llm, ep, prev=None, nxt=None):
    user = build_prompt(ep, prev, nxt)
    meta = {"episode": ep.number, "events": [e.__dict__ for e in ep.events]}
    obj, usage, cached = llm.call_json("propose", SYSTEM, user, meta=meta, episode=ep.number)
    out = []
    for i, r in enumerate(obj.get("promos", []) if isinstance(obj, dict) else [], 1):
        segs = []
        for s in (r.get("segments") or []):
            if isinstance(s, dict) and s.get("start_event") and s.get("end_event"):
                segs.append({"episode": ep.number, "start_event": str(s["start_event"]), "end_event": str(s["end_event"])})
        out.append({
            "source": "ai", "shape": str(r.get("shape") or ""), "shape_reason": str(r.get("shape_reason") or ""),
            "segments": segs, "hook_line": str(r.get("hook_line") or ""),
            "cliffhanger_line": str(r.get("cliffhanger_line") or ""), "why": str(r.get("why") or ""),
            "new_viewer_needs": str(r.get("new_viewer_needs") or ""), "ai_rank": i,
        })
    return out, cached
