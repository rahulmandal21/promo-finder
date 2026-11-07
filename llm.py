"""Provider-agnostic LLM client. Provider/model come ONLY from env vars (see config.py).
Strict-JSON replies, raw-reply cache, 429/5xx retry. Errors never contain the URL or API key."""
import hashlib
import json
import re
import time
from pathlib import Path

import requests

import config


class LLMError(Exception):
    pass


def _scrub(msg, secrets=()):
    msg = str(msg)
    for s in secrets:
        if s:
            msg = msg.replace(s, "***")
    msg = re.sub(r"https?://\S+", "<url>", msg)
    msg = re.sub(r"(AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{10,})", "***", msg)
    return msg[:300]


def _msg(r):
    try:
        d = r.json()
        d = d.get("error", d)
        return d.get("message", d) if isinstance(d, dict) else d
    except Exception:
        return r.text


def _post(url, headers, payload, secrets):
    n429 = n5xx = 0
    while True:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=config.TIMEOUT)
        except requests.Timeout:
            r = None
            code = "timeout"
        except requests.RequestException as e:
            raise LLMError(f"network error ({type(e).__name__})") from None
        if r is not None and r.status_code == 429:
            body = (r.text or "")
            low = body.lower()
            if "per day" in low or "perday" in low:   # daily quota: waiting a minute will not help
                raise LLMError("DAILY free-tier quota used up for this model. Google says: "
                               + _scrub(_msg(r), secrets)) from None
            n429 += 1
            if n429 > config.MAX_429_RETRIES:
                raise LLMError(f"rate limited (HTTP 429) after {config.MAX_429_RETRIES} retries. Google says: "
                               + _scrub(_msg(r), secrets)) from None
            print(f"  ...rate limited (429), waiting {config.WAIT_429:.0f}s then retrying "
                  f"({n429}/{config.MAX_429_RETRIES})", flush=True)
            time.sleep(config.WAIT_429)
            continue
        if r is None or r.status_code in (500, 502, 503, 504):
            n5xx += 1
            if n5xx > config.MAX_5XX_RETRIES:
                raise LLMError(f"server error ({code if r is None else r.status_code}) after retries") from None
            time.sleep(min(config.BACKOFF_BASE * 2 ** (n5xx - 1), 60))
            continue
        if r.status_code >= 400:
            try:
                detail = r.json()
                detail = detail.get("error", detail)
                detail = detail.get("message", detail) if isinstance(detail, dict) else detail
            except Exception:
                detail = r.text
            raise LLMError(f"HTTP {r.status_code}: {_scrub(detail, secrets)}") from None
        try:
            return r.json()
        except ValueError:
            raise LLMError("provider returned a non-JSON HTTP body") from None


def _gemini(model, system, user, key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
    }
    d = _post(url, {"x-goog-api-key": key, "Content-Type": "application/json"}, payload, [key])
    cands = d.get("candidates") or []
    if not cands:
        raise LLMError(f"empty response ({_scrub(d.get('promptFeedback', ''), [key])})")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text:
        raise LLMError(f"empty response (finish reason {cands[0].get('finishReason')})")
    u = d.get("usageMetadata") or {}
    return text, {"in": u.get("promptTokenCount", 0),
                  "out": (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0)}


def _openai(model, system, user, key):
    import os
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {"model": model, "temperature": 0.4,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if os.getenv("PROMO_JSON_MODE", "1") != "0":
        payload["response_format"] = {"type": "json_object"}
    d = _post(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"}, payload, [key])
    try:
        text = d["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected response shape") from None
    u = d.get("usage") or {}
    return text, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}


def _anthropic(model, system, user, key):
    payload = {"model": model, "max_tokens": 4096, "temperature": 0.4, "system": system,
               "messages": [{"role": "user", "content": user}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    d = _post("https://api.anthropic.com/v1/messages", headers, payload, [key])
    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    if not text:
        raise LLMError("empty response")
    u = d.get("usage") or {}
    return text, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}


def _parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except ValueError:
        raise ValueError("invalid JSON") from None
    # some models wrap the one object in a list: [ {...} ]  -> unwrap it
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]
    return obj


class LLM:
    def __init__(self, provider=None, model=None, api_key=None, cache_dir=None):
        self.provider = (provider or config.provider()).lower()
        if self.provider not in config.DEFAULT_MODELS:
            raise LLMError(f"unknown provider '{self.provider}' (use gemini, openai, anthropic or mock)")
        self.model = model or config.model(self.provider)
        self._key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.live_calls = 0
        self.cached_calls = 0

    def _api_key(self):
        k = self._key or config.api_key(self.provider)
        if not k:
            raise LLMError(f"No API key for '{self.provider}'. Set {config.KEY_ENV[self.provider][0]} "
                           f"(environment, .env or Streamlit secrets).")
        return k

    def _raw(self, task, system, user, meta):
        if self.provider == "mock":
            from mock_provider import mock_call
            return mock_call(task, system, user, meta or {})
        key = self._api_key()
        fn = {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[self.provider]
        return fn(self.model, system, user, key)

    def _ck(self, task, system, user):
        return hashlib.sha256("\x1f".join([self.provider, self.model, task, system, user]).encode()).hexdigest()[:20]

    def call_json(self, task, system, user, meta=None, episode=None):
        """-> (parsed_json, usage, was_cached). Retries invalid JSON once. Caches the raw reply."""
        key = self._ck(task, system, user)
        f = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if f and f.exists():
            try:
                c = json.loads(f.read_text())
                self.cached_calls += 1
                return _parse_json(c["text"]), c.get("usage", {}), True
            except (ValueError, KeyError):
                pass
        text, usage = self._raw(task, system, user, meta)
        self.live_calls += 1
        try:
            obj = _parse_json(text)
        except ValueError:
            text, u2 = self._raw(task, system, user + "\n\nYour previous reply was not valid JSON. "
                                 "Reply again with ONLY one valid JSON object, nothing else.", meta)
            self.live_calls += 1
            usage = {"in": usage.get("in", 0) + u2.get("in", 0), "out": usage.get("out", 0) + u2.get("out", 0)}
            try:
                obj = _parse_json(text)
            except ValueError:
                raise LLMError("model returned invalid JSON twice") from None
        if f:
            f.write_text(json.dumps({"provider": self.provider, "model": self.model, "task": task,
                                     "episode": episode, "usage": usage, "text": text}))
        return obj, usage, False
