"""Streamlit front end: upload episodes -> ranked promos. streamlit run app.py"""
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

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebar"] {background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);}
[data-testid="stSidebar"] * {color: #e0e0e0 !important;}
div[data-testid="stMetric"] {background: #1e1e2f; border-radius: 12px; padding: 16px; border: 1px solid #333;}
div[data-testid="stMetric"] label {color: #aaa !important; font-size: 0.85rem;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: #fff !important; font-size: 1.8rem;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 600;}
div[data-testid="stExpander"] {border: 1px solid #333; border-radius: 10px; margin-bottom: 8px;}
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

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()
    st.markdown("#### 🤖 Model")
    d_prov = config.provider() if config.provider() in PROVIDERS else "gemini"
    provider = st.selectbox("Provider", PROVIDERS, index=PROVIDERS.index(d_prov), label_visibility="collapsed")
    model = st.text_input("Model name", value=config.model(provider) or config.DEFAULT_MODELS[provider])
    key_box = st.text_input("🔑 API key (optional)", type="password",
                            help="Leave empty to use the key from environment variables or Streamlit secrets.")
    st.divider()
    st.markdown("#### 🎯 Run settings")
    blind_k = st.slider("Blind viewer clips per episode", 0, 8, config.BLIND_TOP_K)
    delay = st.number_input("Delay between episodes (s)", 0.0, 60.0, float(config.delay()))
    if provider == "mock":
        st.warning("⚠️ Mock provider: offline test heuristics, not a real model.")
    st.divider()
    st.caption("📁 Uploaded files stay in memory for this session only.")

# ---- header ----
st.markdown("""
# 🎬 Promo Finder
**Find the best 20–60s promo clips from AI microdrama screenplays.**

Each promo opens on a hook, ends on a cliffhanger, and makes sense to someone who has never seen the show.
Upload episode scripts below, pick a model, and click **Run**.
""")

col_up, col_info = st.columns([3, 1])
with col_up:
    files = st.file_uploader("📄 Episode scripts (.txt)", type=["txt"], accept_multiple_files=True,
                             label_visibility="collapsed")
with col_info:
    if files:
        st.success(f"**{len(files)}** file{'s' if len(files) != 1 else ''} uploaded")
    else:
        st.info("Upload .txt episode files to start")

run = st.button("🚀 Run analysis", type="primary", disabled=not files, use_container_width=True)

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
        st.warning("⚠️ Input notes:\n\n" + "\n".join(f"- {p}" for p in problems))
    if not episodes:
        st.error("❌ No readable episodes found.")
        st.stop()
    if "cache" not in st.session_state:
        st.session_state["cache"] = tempfile.mkdtemp(prefix="promo_cache_")
    try:
        llm = LLM(provider, model, api_key=key_box or None, cache_dir=st.session_state["cache"])
        llm._api_key() if provider != "mock" else None
    except LLMError as e:
        st.error(f"❌ {e}")
        st.stop()
    bar = st.progress(0.0, text="⏳ Starting...")
    prog = lambda i, n, e: bar.progress(i / max(n, 1), text=f"⏳ Processing episode {e}..." if e else "✅ Finishing...")
    with st.spinner("Working..."):
        st.session_state["res"] = run_all.process(episodes, llm, blind_k=blind_k, delay=delay, progress=prog)
    bar.empty()
    st.balloons()

res = st.session_state.get("res")
if res:
    picks = res["picks"]

    # ---- metrics row ----
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("✅ Episodes", len(res["ep_stats"]))
    c2.metric("🏆 Top picks", len(picks))
    c3.metric("🛡️ Rejected", sum(v["rejected"] for v in res["ep_stats"].values()))
    c4.metric("✂️ Mid-line cuts", res["midline_cuts"])
    c5.metric("🤖 AI promos", sum(v["ai_proposed"] for v in res["ep_stats"].values()))

    if res["errors"]:
        st.markdown("---")
        for n, msg in sorted(res["errors"].items()):
            st.error(f"Episode {n}: {msg}")
        if res.get("aborted"):
            st.error(f"⛔ {res['aborted']}")

    if provider == "mock":
        st.warning("⚠️ **MOCK RUN** — scores are test heuristics, not a real model's judgement.")

    # ---- tabs ----
    t_rank, t_post, t_fail, t_dl = st.tabs(["📊 Ranked Promos", "📱 Post First", "⚠️ Failures", "💾 Download"])

    with t_rank:
        st.markdown("### Top promo picks across all episodes")
        df = pd.DataFrame([{
            "Rank": p["rank"],
            "Episode": ",".join(map(str, episodes_of(p))),
            "Timestamps": segs_str(p),
            "Shape": p["shape"],
            "Source": "🟦 code" if p["source"] == "code" else "🟪 ai",
            "Score": round(p["rank_score"], 2),
            "Hook": p["scores"]["hook"],
            "Cliff": p["scores"]["cliffhanger"],
            "Standalone": round(p["scores"].get("standalone_final", p["scores"]["standalone"]), 1),
            "Blind OK": "—" if not p.get("blind") else ("✅" if p["blind"]["ok"] else "❌"),
            "Verified": "✅" if p["verified"] else "❌",
        } for p in picks])
        st.dataframe(df, hide_index=True, width="stretch")

        st.markdown("### Promo details")
        for p in picks[:20]:
            ep_label = ",".join(map(str, episodes_of(p)))
            badge = "🟦 Code" if p["source"] == "code" else "🟪 AI"
            score = p["rank_score"]
            with st.expander(f"**#{p['rank']}** — Ep {ep_label} — {segs_str(p)} — {badge} — Score {score:.1f}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**🎣 Hook:** {p['hook_line']}")
                    st.markdown(f"**🔥 Cliffhanger:** {p['cliffhanger_line']}")
                    st.markdown(f"**💡 Why:** {p.get('why') or p['rank_reason']}")
                with col_b:
                    st.markdown(f"**📐 Shape:** `{p['shape']}`")
                    st.markdown(f"**⏱️ Duration:** {p.get('duration', 0):.0f}s")
                    st.markdown(f"**🏅 Judge:** {p['rank_reason']}")
                if p.get("blind"):
                    b = p["blind"]
                    status = "✅ Understood" if b["ok"] else "❌ Confused"
                    st.markdown(f"**👁️ Blind viewer** ({status}): *{b['who']}* → wants to know: *{b['want_to_know']}*"
                                + (f" — confusion: *{b['confusion']}*" if b["confusion"] else ""))
                if p.get("penalties") or p.get("flags"):
                    parts = []
                    if p.get("penalties"):
                        parts.append("Penalties: " + ", ".join(f"{k} (−{v})" for k, v in p["penalties"].items()))
                    if p.get("flags"):
                        parts.append("Flags: " + ", ".join(p["flags"]))
                    st.caption(" | ".join(parts))

    with t_post:
        st.markdown(post_first.to_markdown(res["post_first"], res["skipped"], res["provider"]))

    with t_fail:
        import failures as F
        st.markdown(F.build(res["all"], picks, res["errors"], res["ep_stats"], res["bstats"],
                            res["provider"], res["model"]))

    with t_dl:
        st.markdown("### Download raw outputs")
        col_d1, col_d2 = st.columns(2)
        buf = io.StringIO()
        run_all.write_csv(picks, buf)
        clean = [{k: v for k, v in p.items() if not k.startswith("_")} for p in res["all"]]
        with col_d1:
            st.download_button("📥 promos_ranked.csv", buf.getvalue(), "promos_ranked.csv", "text/csv",
                               use_container_width=True)
        with col_d2:
            st.download_button("📥 promos_all.json",
                               json.dumps({"provider": res["provider"], "model": res["model"],
                                           "promos": clean}, indent=1, ensure_ascii=False),
                               "promos_all.json", "application/json", use_container_width=True)
