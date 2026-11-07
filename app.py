"""Streamlit front end: upload episodes -> ranked promos. streamlit run app.py"""
import io
import json
import os
import tempfile

import pandas as pd
import streamlit as st

import config
import failures  # noqa: F401  (imported so a missing file fails loudly at start-up)
import post_first
import run_all
from llm import LLM, LLMError
from parser import parse_episode_text
from utils import episodes_of, segs_str

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")


def _load_secrets():
    """Streamlit secrets -> env vars (env vars still win). Safe when no secrets file exists."""
    names = ["PROMO_PROVIDER", "PROMO_MODEL", "PROMO_DELAY", "OPENAI_BASE_URL", "APP_PASSWORD"] + sum(config.KEY_ENV.values(), [])
    for k in names:
        try:
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
        except Exception:
            return


_load_secrets()

if os.getenv("APP_PASSWORD"):   # optional gate so strangers cannot spend your API quota
    if st.text_input("Password", type="password") != os.environ["APP_PASSWORD"]:
        st.info("Enter the password to use this app.")
        st.stop()

PROVIDERS = ["gemini", "openai", "anthropic", "mock"]

with st.sidebar:
    st.header("Model")
    d_prov = config.provider() if config.provider() in PROVIDERS else "gemini"
    provider = st.selectbox("Provider", PROVIDERS, index=PROVIDERS.index(d_prov))
    model = st.text_input("Model", value=config.model(provider) or config.DEFAULT_MODELS[provider])
    key_box = st.text_input("API key (optional)", type="password",
                            help="Leave empty to use the key from environment variables or Streamlit secrets.")
    st.header("Run")
    blind_k = st.slider("Clips per episode sent to the blind viewer", 0, 8, config.BLIND_TOP_K)
    delay = st.number_input("Seconds between episodes", 0.0, 60.0, float(config.delay()))
    if provider == "mock":
        st.warning("Mock provider: offline test heuristics, not a real model. Scores are meaningless.")
    st.caption("Uploaded files stay in memory for this session and are never written to disk.")

st.title("🎬 Promo Finder")
st.write("Upload microdrama episode scripts (.txt). The tool proposes 20-60 s promo clips with a hook, a "
         "cliffhanger and exact timestamps, checks them with code guards and a blind viewer, and ranks them.")

files = st.file_uploader("Episode scripts", type=["txt"], accept_multiple_files=True)
run = st.button("Run", type="primary", disabled=not files)

if run:
    episodes, problems = {}, []
    for f in files:
        try:
            ep = parse_episode_text(f.getvalue().decode("utf-8", errors="replace"), f.name)
            if ep.number in episodes:
                problems.append(f"{f.name}: duplicate episode number {ep.number}; later file used")
            episodes[ep.number] = ep
            problems += [f"{f.name}: {w}" for w in ep.warnings]
        except ValueError as e:
            problems.append(f"{f.name}: {e}")
    if problems:
        st.warning("Input notes:\n\n" + "\n".join(f"- {p}" for p in problems))
    if not episodes:
        st.error("No readable episodes.")
        st.stop()
    if "cache" not in st.session_state:
        st.session_state["cache"] = tempfile.mkdtemp(prefix="promo_cache_")
    try:
        llm = LLM(provider, model, api_key=key_box or None, cache_dir=st.session_state["cache"])
        llm._api_key() if provider != "mock" else None
    except LLMError as e:
        st.error(str(e))
        st.stop()
    bar = st.progress(0.0, text="Starting")
    prog = lambda i, n, e: bar.progress(i / max(n, 1), text=f"Episode {e}" if e else "Finishing")
    with st.spinner("Working..."):
        st.session_state["res"] = run_all.process(episodes, llm, blind_k=blind_k, delay=delay, progress=prog)
    bar.empty()

res = st.session_state.get("res")
if res:
    picks = res["picks"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes done", len(res["ep_stats"]), f"{len(res['errors'])} failed" if res["errors"] else None,
              delta_color="inverse")
    c2.metric("Top picks", len(picks))
    c3.metric("Rejected by guards", sum(v["rejected"] for v in res["ep_stats"].values()))
    c4.metric("Mid-line cuts", res["midline_cuts"])
    if res["provider"] == "mock":
        st.warning("MOCK RUN - scores are test heuristics, not a real model's judgement.")
    for n, msg in sorted(res["errors"].items()):
        st.error(f"Episode {n}: {msg}  (re-run: finished calls are cached)")
    if res.get("aborted"):
        st.error(res["aborted"])

    t_rank, t_post, t_fail, t_dl = st.tabs(["Ranked promos", "Post first", "Failures", "Download"])
    with t_rank:
        df = pd.DataFrame([{
            "Rank": p["rank"], "Ep": ",".join(map(str, episodes_of(p))), "Segments": segs_str(p),
            "Shape": p["shape"], "Source": ("🟦 code" if p["source"] == "code" else "🟪 ai"),
            "Score": round(p["rank_score"], 2), "Hook": p["scores"]["hook"], "Cliff": p["scores"]["cliffhanger"],
            "Standalone": round(p["scores"].get("standalone_final", p["scores"]["standalone"]), 1),
            "Blind OK": "n/a" if not p.get("blind") else ("yes" if p["blind"]["ok"] else "no"),
            "Verified": "yes" if p["verified"] else "no",
        } for p in picks])
        st.dataframe(df, hide_index=True, width="stretch")
        for p in picks[:15]:
            with st.expander(f"#{p['rank']}  Ep {','.join(map(str, episodes_of(p)))}  -  {segs_str(p)}"):
                st.markdown(f"**Hook:** {p['hook_line']}  \n**Cliffhanger:** {p['cliffhanger_line']}  \n"
                            f"**Why:** {p.get('why') or p['rank_reason']}  \n**Judge:** {p['rank_reason']}")
                if p.get("blind"):
                    b = p["blind"]
                    st.markdown(f"**Blind viewer** ({'understood' if b['ok'] else 'confused'}): who: {b['who']} "
                                f"| next: {b['want_to_know']}" + (f" | confusion: {b['confusion']}" if b["confusion"] else ""))
                if p.get("penalties") or p.get("flags") or p.get("repairs"):
                    st.caption(f"penalties: {p.get('penalties')}  flags: {p.get('flags')}  repairs: {sorted(set(p.get('repairs', [])))}")
    with t_post:
        st.markdown(post_first.to_markdown(res["post_first"], res["skipped"], res["provider"]))
    with t_fail:
        import failures as F
        st.markdown(F.build(res["all"], picks, res["errors"], res["ep_stats"], res["bstats"], res["provider"], res["model"]))
    with t_dl:
        buf = io.StringIO()
        run_all.write_csv(picks, buf)
        clean = [{k: v for k, v in p.items() if not k.startswith("_")} for p in res["all"]]
        st.download_button("promos_ranked.csv", buf.getvalue(), "promos_ranked.csv", "text/csv")
        st.download_button("promos_all.json", json.dumps({"provider": res["provider"], "model": res["model"],
                           "promos": clean}, indent=1, ensure_ascii=False), "promos_all.json", "application/json")
