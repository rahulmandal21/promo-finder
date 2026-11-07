"""Screenplay .txt -> timed events. Plain code, no AI. Parsing is case-insensitive."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import config

HEADER_RE = re.compile(r"^\*\*(.+?)\*\*\s*:?\s*$")
TS_RE = re.compile(r"Start:\s*\(?(\d+):(\d+(?:\.\d+)?)\)?\s*\|\s*End:\s*\(?(\d+):(\d+(?:\.\d+)?)\)?", re.I)
SCENE_RE = re.compile(r"^SCENE\s*(\d+)\s*$", re.I)
STOP_RE = re.compile(r"^(---|Scene heading:|Scene start:|Scene end:|Actions and dialogue:)", re.I)


@dataclass
class Event:
    id: str
    type: str            # camera | action | sound | dialogue
    speaker: str | None
    tone: str | None
    text: str
    start: float
    end: float
    scene: int = 0

    @property
    def is_cue(self):    # writers' own "cliffhanger" note in a SOUND FX line
        return self.type == "sound" and "cliffhanger" in self.text.lower()


@dataclass
class Episode:
    number: int
    logline: str
    continuity: str
    events: list
    source_file: str = ""
    warnings: list = field(default_factory=list)

    def __post_init__(self):
        self.by_id = {e.id: e for e in self.events}
        self.dialogue = [e for e in self.events if e.type == "dialogue"]
        self.duration = max((e.end for e in self.events), default=0.0)

    # --- spoken-line safety -------------------------------------------------
    def inside_dialogue(self, t, tol=config.TOL):
        for d in self.dialogue:
            if d.start + tol < t < d.end - tol:
                return d
        return None

    def snap_start(self, t):
        hit = None
        for _ in range(6):
            d = self.inside_dialogue(t)
            if not d:
                break
            t, hit = d.start, d
        return t, hit

    def snap_end(self, t):
        hit = None
        for _ in range(6):
            d = self.inside_dialogue(t)
            if not d:
                break
            t, hit = d.end, d
        return t, hit

    def events_in(self, start, end):
        return [e for e in self.events if start - config.TOL <= e.start < end - config.TOL]


def _num(s):
    return float(s)


def parse_episode_text(text, filename=""):
    warns = []
    m = re.search(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", text, re.I)
    if m:
        number = int(m.group(1))
    else:
        f = re.search(r"episode[-_ ]?(\d+)", filename, re.I)
        if not f:
            raise ValueError(f"No EPISODE NUMBER found in {filename or 'text'}")
        number = int(f.group(1))
        warns.append("EPISODE NUMBER line missing; used filename")
    logline = (re.search(r"EPISODE LOGLINE:\s*(.*)", text, re.I) or [None, ""])[1].strip()
    cm = re.search(r"CONTINUITY NOTES:\s*(.*?)(?:\n\s*\n|\n---)", text, re.I | re.S)
    continuity = cm.group(1).strip() if cm else ""

    raw, cur, scene = [], None, 0

    def close():
        nonlocal cur
        if cur is not None:
            raw.append(cur)
        cur = None

    for line in text.splitlines():
        s = line.strip()
        sm = SCENE_RE.match(s)
        if sm:
            close()
            scene = int(sm.group(1))
            continue
        if STOP_RE.match(s):
            close()
            continue
        h = HEADER_RE.match(s)
        if h:
            close()
            cur = {"label": h.group(1).strip().rstrip(":").strip(), "ts": None, "body": [], "scene": scene}
            continue
        if cur is None:
            continue
        t = TS_RE.search(s)
        if t and cur["ts"] is None:
            a, b, c, d = t.groups()
            cur["ts"] = (int(a) * 60 + _num(b), int(c) * 60 + _num(d))
        elif s:
            cur["body"].append(s)
    close()

    events = []
    for i, r in enumerate(raw):
        if r["ts"] is None:
            warns.append(f"block '{r['label']}' has no timestamps; skipped")
            continue
        lab = r["label"]
        up = lab.upper()
        speaker = tone = None
        if up.startswith("CAMERA"):
            typ = "camera"
        elif up.startswith("ACTION"):
            typ = "action"
        elif up.startswith("SOUND") or up.startswith("SFX"):
            typ = "sound"
        else:
            typ = "dialogue"
            mm = re.match(r"^(.*?)\s*\((.*)\)\s*$", lab)
            speaker, tone = (mm.group(1).strip(), mm.group(2).strip()) if mm else (lab.strip(), "")
        body = " ".join(r["body"]).strip()
        if typ == "dialogue":
            body = body.strip().strip('"\u201c\u201d').strip()
        if not body:
            continue
        s0, e0 = r["ts"]
        if e0 < s0:
            warns.append(f"block '{lab}' end before start; skipped")
            continue
        events.append((s0, e0, i, typ, speaker, tone, body, r["scene"]))
    events.sort(key=lambda x: (x[0], x[1], x[2]))
    evs = [
        Event(f"E{number}-{k:03d}", typ, sp, tn, body, s0, e0, sc)
        for k, (s0, e0, _, typ, sp, tn, body, sc) in enumerate(events, 1)
    ]
    if not any(e.type == "dialogue" for e in evs):
        warns.append("no dialogue found")
    return Episode(number, logline, continuity, evs, filename, warns)


def load_dir(path):
    eps, notes = {}, []
    for f in sorted(Path(path).glob("*.txt")):
        try:
            ep = parse_episode_text(f.read_text(encoding="utf-8", errors="replace"), f.name)
        except ValueError as e:
            notes.append(str(e))
            continue
        if ep.number in eps:
            notes.append(f"duplicate episode number {ep.number} ({f.name}); later file wins")
        eps[ep.number] = ep
    return eps, notes
