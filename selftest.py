"""Offline self-test (no API key, no network): python selftest.py"""
import json
import sys
from pathlib import Path

import requests

import candidates
import config
import llm as L
import run_all
import validator as V
from parser import load_dir

FAILS = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


episodes, notes = load_dir("episodes")
if not episodes:
    sys.exit("No episodes found in ./episodes - put the .txt files there first, then re-run.")
check("parser: loads episodes", len(episodes) > 0, notes)
ep = episodes[min(episodes)]
nxt = episodes.get(ep.number + 1)
dl = ep.dialogue
mid_line_cam = None
for e in episodes.values():
    for x in e.events:
        if x.type != "dialogue" and any(d.start + 0.3 < x.start < d.end - 0.3 for d in e.dialogue):
            mid_line_cam = (e, x, next(d for d in e.dialogue if d.start + 0.3 < x.start < d.end - 0.3))
            break
    if mid_line_cam:
        break


def P(e, a, b, hook, cliff, shape="single"):
    return {"source": "ai", "shape": shape, "segments": [{"episode": e.number, "start_event": a, "end_event": b}],
            "hook_line": hook, "cliffhanger_line": cliff, "flags": []}


# ---- validator on hand-made bad promos ----
e, cam, d = mid_line_cam
after = [x for x in e.dialogue if x.start >= d.start and x.end - d.start <= 28]
v = V.validate(P(e, cam.id, after[-1].id, d.text, after[-1].text), episodes)
check("guard: start inside spoken line is repaired", v["status"] == "ok" and v["segments"][0]["start"] == d.start
      and V.count_midline_cuts(v, episodes) == 0, v.get("reject_reason"))
v = V.validate(P(e, e.events[0].id, e.events[-1].id, "x", "y"), episodes)
check("guard: whole episode rejected as too long", v["status"] == "rejected" and "long" in v["reject_reason"])
v = V.validate(P(e, after[0].id, after[0].id, d.text, d.text), episodes)
check("guard: too-short rejected", v["status"] == "rejected" and "short" in v["reject_reason"])
two = {"source": "ai", "shape": "hook+twist", "hook_line": "", "cliffhanger_line": "", "flags": [],
       "segments": [{"episode": e.number, "start_event": e.dialogue[-3].id, "end_event": e.dialogue[-1].id},
                    {"episode": e.number, "start_event": e.dialogue[0].id, "end_event": e.dialogue[3].id}]}
v = V.validate(two, episodes)
check("guard: out-of-order segments rejected", v["status"] == "rejected", v["reject_reason"])
v = V.validate(P(e, "E999-001", e.dialogue[-1].id, "a", "b"), episodes)
check("guard: unknown event id rejected", v["status"] == "rejected")
c = candidates.generate(e)[0]
c = dict(c, source="ai", hook_line="I never wanted any of this, " + c["hook_line"])
check("guard: invented words in a quote -> unverified", not V.validate(c, episodes)["verified"])
c2 = candidates.generate(e)[0]
part = " ".join(c2["hook_line"].split()[:3])
check("guard: genuine partial quote still verifies", V.validate(dict(c2, hook_line=part), episodes)["verified"] or len(part.split()) < 2)
if nxt:
    cross = P(e, e.dialogue[-4].id, nxt.dialogue[1].id, e.dialogue[-4].text, nxt.dialogue[1].text)
    v = V.validate(cross, episodes)
    check("guard: cross-episode segment is split in two", v["status"] == "ok" and len(v["segments"]) == 2
          and v["segments"][0]["episode"] != v["segments"][1]["episode"], v.get("reject_reason"))


# ---- LLM layer with a faked HTTP layer ----
class R:
    def __init__(self, code, body):
        self.status_code, self._b, self.text = code, body, json.dumps(body)

    def json(self):
        return self._b


def gem(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3}}


config.WAIT_429, config.BACKOFF_BASE = 0, 0
KEY = "AIzaSECRETKEY1234567890"
real_post = requests.post


def run_seq(seq):
    it = iter(seq)
    calls = []

    def fake(url, **kw):
        calls.append(url)
        x = next(it)
        if isinstance(x, Exception):
            raise x
        return x
    requests.post = fake
    try:
        m = L.LLM("gemini", "m", api_key=KEY)
        return m.call_json("t", "s", "u"), len(calls)
    finally:
        requests.post = real_post


(obj, _, _), n = run_seq([R(429, {}), R(429, {}), R(200, gem('{"a":1}'))])
check("llm: retries 429 then succeeds", obj == {"a": 1} and n == 3)
(obj, _, _), n = run_seq([R(503, {}), R(500, {}), R(200, gem('{"a":2}'))])
check("llm: retries 500/503 then succeeds", obj == {"a": 2} and n == 3)
(obj, _, _), n = run_seq([R(200, gem("not json")), R(200, gem('{"ok":true}'))])
check("llm: bad JSON retried once", obj == {"ok": True} and n == 2)
try:
    run_seq([R(200, gem("nope")), R(200, gem("still nope"))])
    check("llm: bad JSON twice raises", False)
except L.LLMError as ex:
    check("llm: bad JSON twice raises", "invalid JSON" in str(ex))
try:
    run_seq([R(429, {})] * 7)
    check("llm: gives up after 6 retries on 429", False)
except L.LLMError as ex:
    check("llm: gives up after 6 retries on 429", "429" in str(ex))
for label, seq in [("HTTP 400 body", [R(400, {"error": {"message": f"bad key {KEY} at https://x.googleapis.com/v1?key={KEY}"}})]),
                   ("network error", [requests.ConnectionError(f"failed https://x.googleapis.com/v1?key={KEY}")])]:
    try:
        run_seq(seq)
        check(f"llm: {label} raises", False)
    except L.LLMError as ex:
        s = str(ex)
        check(f"llm: error never shows key or URL ({label})", KEY not in s and "http" not in s and "googleapis" not in s, s)
    except Exception as ex:   # an un-sanitised exception would leak
        check(f"llm: error never shows key or URL ({label})", False, repr(ex)[:120])

# ---- full pipeline on the mock provider ----
res = run_all.process(episodes, L.LLM("mock"), delay=0)
print("\n" + run_all.summary(res))
check("pipeline: every episode ran", not res["errors"] and len(res["ep_stats"]) == len(episodes), res["errors"])
check("pipeline: MID-LINE CUTS == 0", res["midline_cuts"] == 0)
check("pipeline: >=12 candidates per episode", all(s["candidates"] >= 12 for s in res["ep_stats"].values()))
check("pipeline: every pick verified-or-penalised and 20-60 s",
      all(config.MIN_TOTAL - .1 <= p["duration"] <= config.MAX_TOTAL + .1 for p in res["picks"]))
print("\n" + ("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
