"""Streamlit front end for Promo Finder — Kuku-styled."""
import io
import json
import os
import tempfile

import pandas as pd
import streamlit as st

import config
import failures
import post_first
import run_all
from llm import LLM, LLMError
from parser import parse_episode_text
from utils import episodes_of, segs_str

st.set_page_config(page_title="Promo Finder · Kuku", page_icon="🔴", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ---- Kuku palette: dark bg, red accent ---- */
:root {
    --kuku-red: #E53935;
    --kuku-dark: #0d0d0d;
    --kuku-card: #1a1a1a;
    --kuku-border: #2a2a2a;
    --kuku-text: #e8e8e8;
    --kuku-muted: #888;
}

/* sidebar */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0d0d0d 0%, #1a0a0a 100%) !important; border-right: 1px solid #2a1a1a;}
[data-testid="stSidebar"] * {color: var(--kuku-text) !important;}
[data-testid="stSidebar"] .stSelectbox > div > div {background: #1a1a1a; border-color: #333;}
[data-testid="stSidebar"] input {background: #1a1a1a !important; border-color: #333 !important;}
[data-testid="stSidebar"] .stDivider {border-color: #2a1a1a !important;}

/* metrics */
div[data-testid="stMetric"] {
    background: var(--kuku-card); border-radius: 12px; padding: 18px 16px;
    border: 1px solid var(--kuku-border); box-shadow: 0 2px 8px rgba(229,57,53,0.08);
}
div[data-testid="stMetric"] label {color: var(--kuku-muted) !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: #fff !important; font-size: 2rem; font-weight: 700;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 2px solid var(--kuku-border);}
.stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0; padding: 10px 24px; font-weight: 600; color: var(--kuku-muted);}
.stTabs [data-baseweb="tab"][aria-selected="true"] {color: var(--kuku-red) !important; border-bottom: 3px solid var(--kuku-red);}

/* expanders */
div[data-testid="stExpander"] {
    background: var(--kuku-card); border: 1px solid var(--kuku-border); border-radius: 10px;
    margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
div[data-testid="stExpander"]:hover {border-color: var(--kuku-red); box-shadow: 0 2px 12px rgba(229,57,53,0.15);}

/* primary button */
.stButton > button[kind="primary"] {
    background: var(--kuku-red) !important; color: #fff !important; border: none !important;
    border-radius: 8px; font-weight: 700; font-size: 1rem; padding: 0.6rem 2rem;
    transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover {background: #c62828 !important; box-shadow: 0 4px 16px rgba(229,57,53,0.4);}

/* download buttons */
.stDownloadButton > button {
    background: transparent !important; border: 1px solid var(--kuku-red) !important;
    color: var(--kuku-red) !important; border-radius: 8px; font-weight: 600;
}
.stDownloadButton > button:hover {background: rgba(229,57,53,0.1) !important;}

/* file uploader */
[data-testid="stFileUploader"] {border: 2px dashed var(--kuku-border); border-radius: 12px; padding: 16px;}
[data-testid="stFileUploader"]:hover {border-color: var(--kuku-red);}

/* dataframe */
[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}

/* divider */
hr {border-color: var(--kuku-border) !important;}
</style>
""", unsafe_allow_html=True)


def _load_secrets():
    names = ["PROMO_PROVIDER", "PROMO_MODEL", "PROMO_DELAY", "OPENAI_BASE_URL"] + sum(config.KEY_ENV.values(), [])
    for k in names:
        try:
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
        except Exception:
            return


_load_secrets()
PROVIDERS = ["gemini", "openai", "anthropic", "mock"]

# ---- sidebar ----
with st.sidebar:
    st.markdown('<h2 style="color:#E53935;margin:0;">🔴 Promo Finder</h2>', unsafe_allow_html=True)
    st.caption("by Kuku · AI Microdrama Tools")
    st.divider()
    st.markdown("**Model**")
    d_prov = config.provider() if config.provider() in PROVIDERS else "gemini"
    provider = st.selectbox("Provider", PROVIDERS, index=PROVIDERS.index(d_prov), label_visibility="collapsed")
    model = st.text_input("Model", value=config.model(provider) or config.DEFAULT_MODELS[provider])
    key_box = st.text_input("API key (optional)", type="password",
                            help="Leave empty to use the key from env vars or Streamlit secrets.")
    st.divider()
    st.markdown("**Run settings**")
    blind_k = st.slider("Blind viewer clips / episode", 0, 8, config.BLIND_TOP_K)
    delay = st.number_input("Delay between episodes (s)", 0.0, 60.0, float(config.delay()))
    if provider == "mock":
        st.error("Mock mode — scores are fake heuristics")
    st.divider()
    st.caption("Files stay in memory for this session only. Nothing is stored.")

# ---- main area ----
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem;">
    <span style="font-size:3rem;">🎬</span>
    <h1 style="margin:0; font-size:2.4rem; font-weight:800; letter-spacing:-0.5px;">
        Promo Finder
    </h1>
    <p style="color:#888; font-size:1.05rem; margin-top:4px;">
        Find the best 20–60s promo clips from AI microdrama screenplays
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#1a1a1a; border:1px solid #2a2a2a; border-radius:12px; padding:16px 20px; margin:0 0 24px; text-align:center;">
    <span style="color:#E53935; font-weight:700;">How it works:</span>
    <span style="color:#ccc;">
    Upload scripts → AI proposes clips → Code guards verify → Blind viewer tests → Ranked results
    </span>
</div>
""", unsafe_allow_html=True)

files = st.file_uploader("Upload episode scripts (.txt)", type=["txt"], accept_multiple_files=True)

if files:
    st.success(f"**{len(files)}** episode{'s' if len(files) != 1 else ''} ready")

run = st.button("🚀 Run analysis", type="primary", disabled=not files, use_container_width=True)

if run:
    episodes, problems = {}, []
    for f in files:
        try:
            ep = parse_episode_text(f.getvalue().decode("utf-8", errors="replace"), f.name)
            if ep.number in episodes:
                problems.append(f"{f.name}: duplicate episode {ep.number}")
            episodes[ep.number] = ep
            problems += [f"{f.name}: {w}" for w in ep.warnings]
        except ValueError as e:
            problems.append(f"{f.name}: {e}")
    if problems:
        st.warning("⚠️ " + " · ".join(problems))
    if not episodes:
        st.error("No readable episodes found.")
        st.stop()
    if "cache" not in st.session_state:
        st.session_state["cache"] = tempfile.mkdtemp(prefix="promo_cache_")
    try:
        llm = LLM(provider, model, api_key=key_box or None, cache_dir=st.session_state["cache"])
        _ = llm._api_key() if provider != "mock" else None
    except LLMError as e:
        st.error(str(e))
        st.stop()
    bar = st.progress(0.0, text="Starting...")
    prog = lambda i, n, e: bar.progress(i / max(n, 1),
                                         text=f"Processing episode {e}..." if e else "Finishing...")
    with st.spinner(""):
        st.session_state["res"] = run_all.process(episodes, llm, blind_k=blind_k, delay=delay, progress=prog)
    bar.empty()
    st.balloons()

res = st.session_state.get("res")
if res:
    picks = res["picks"]
    st.markdown("---")

    # ---- metrics ----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Episodes", len(res["ep_stats"]))
    c2.metric("Top picks", len(picks))
    c3.metric("Rejected", sum(v["rejected"] for v in res["ep_stats"].values()))
    c4.metric("Mid-line cuts", res["midline_cuts"])
    c5.metric("AI promos", sum(v["ai_proposed"] for v in res["ep_stats"].values()))

    if res["errors"]:
        for n, msg in sorted(res["errors"].items()):
            st.error(f"Episode {n}: {msg}")
        if res.get("aborted"):
            st.error(res["aborted"])

    if provider == "mock":
        st.warning("**MOCK RUN** — scores are test heuristics, not a real model.")

    # ---- tabs ----
    t_rank, t_post, t_fail, t_dl = st.tabs(["📊 Ranked", "📱 Post First", "⚠️ Failures", "💾 Download"])

    with t_rank:
        df = pd.DataFrame([{
            "#": p["rank"],
            "Ep": ",".join(map(str, episodes_of(p))),
            "Timestamps": segs_str(p),
            "Shape": p["shape"],
            "Source": "🟦 code" if p["source"] == "code" else "🟪 ai",
            "Score": round(p["rank_score"], 2),
            "Hook": p["scores"]["hook"],
            "Cliff": p["scores"]["cliffhanger"],
            "Stand": round(p["scores"].get("standalone_final", p["scores"]["standalone"]), 1),
            "Blind": "—" if not p.get("blind") else ("✅" if p["blind"]["ok"] else "❌"),
            "Verified": "✅" if p["verified"] else "❌",
        } for p in picks])
        st.dataframe(df, hide_index=True, width="stretch")

        for p in picks[:20]:
            ep_label = ",".join(map(str, episodes_of(p)))
            badge = "🟦 Code" if p["source"] == "code" else "🟪 AI"
            with st.expander(
                f"**#{p['rank']}** · Ep {ep_label} · {segs_str(p)} · {badge} · {p['rank_score']:.1f}"
            ):
                c_l, c_r = st.columns([3, 2])
                with c_l:
                    st.markdown(f'**Hook:** *"{p["hook_line"]}"*')
                    st.markdown(f'**Cliffhanger:** *"{p["cliffhanger_line"]}"*')
                    st.markdown(f"**Why:** {p.get('why') or p['rank_reason']}")
                with c_r:
                    st.markdown(f"**Shape:** `{p['shape']}`  ·  **Duration:** {p.get('duration', 0):.0f}s")
                    st.markdown(f"**Judge:** {p['rank_reason']}")
                    if p.get("blind"):
                        b = p["blind"]
                        ico = "✅" if b["ok"] else "❌"
                        st.markdown(f"**Blind viewer** {ico}: *{b['who'][:120]}*")
                if p.get("penalties"):
                    st.caption("Penalties: " + ", ".join(f"{k} (−{v})" for k, v in p["penalties"].items()))

    with t_post:
        st.markdown(post_first.to_markdown(res["post_first"], res["skipped"], res["provider"]))

    with t_fail:
        import failures as F
        st.markdown(F.build(res["all"], picks, res["errors"], res["ep_stats"], res["bstats"],
                            res["provider"], res["model"]))

    with t_dl:
        st.markdown("### Export")
        c_d1, c_d2 = st.columns(2)
        buf = io.StringIO()
        run_all.write_csv(picks, buf)
        clean = [{k: v for k, v in p.items() if not k.startswith("_")} for p in res["all"]]
        with c_d1:
            st.download_button("📥 promos_ranked.csv", buf.getvalue(), "promos_ranked.csv", "text/csv",
                               use_container_width=True)
        with c_d2:
            st.download_button("📥 promos_all.json",
                               json.dumps({"provider": res["provider"], "model": res["model"],
                                           "promos": clean}, indent=1, ensure_ascii=False),
                               "promos_all.json", "application/json", use_container_width=True)
