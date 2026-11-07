# promo-finder

Reads AI-microdrama episode scripts (`.txt`) and finds the best 20-60 s promo clips (hook in the first 3 s, unresolved
cliffhanger, understandable to a cold viewer). Only finds moments and exact start/end times: no editing, music or overlays.

**The episodes are confidential.** `episodes/`, `outputs/`, `.env`, `gold.json` are git-ignored. Never commit them.

## How it works

| Step | File | Kind | What it does |
|---|---|---|---|
| 1 | `parser.py` | code | script -> timed events (id, type, speaker, text, start, end) + logline + continuity notes |
| 2 | `candidates.py` | code | 14 safe candidates/episode: opening, ending, hook+twist, many hook+bridge+twist (bridge = most context-rich lines) |
| 3 | `proposer.py` | **AI** | model acts as promo editor, proposes its own promos (any shape; says why if not Hook-Bridge-Twist) |
| 4 | `validator.py` | code | never trusts the model: snaps out of spoken lines, rejects bad length/order/bridge, verifies quoted hook & cliffhanger exist |
| 5 | `scorer.py` | **AI** | one judging call per episode, blind to who proposed each clip (shuffled opaque ids) |
| 6 | `blind_viewer.py` | **AI** | sees ONLY the clip text; "does it make sense on its own?"; tests hook+twist clips with/without a bridge |
| 7 | `ranker.py` | code | 0.35 hook + 0.35 cliffhanger + 0.30 standalone, minus penalties; top 1-3 per episode, no near-duplicates |
| 8 | `post_first.py` | code | the 5 to post first to a cold audience + strong clips skipped for needing context |

The model is chosen **only** by env vars. Swap it without touching code:

```
PROMO_PROVIDER = gemini | openai | anthropic | mock      (default gemini)
PROMO_MODEL    = e.g. gemini-3.5-flash                   (default per provider, see config.py)
PROMO_DELAY    = seconds between episodes                (default 8)
```
Keys: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL` for any OpenAI-compatible API),
`ANTHROPIC_API_KEY`. `mock` needs no key: it is test heuristics, **not** an AI, and its scores mean nothing.

## Run on a Mac (one step at a time)

```bash
cd promo-finder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put the 50 `.txt` files into `./episodes/`, then:
```bash
python selftest.py                                  # offline checks, no key needed
python run_all.py --input ./episodes --out outputs_mock --provider mock     # offline dry run
```
Real run (key in `.env`, see `.env.example`, or `export GEMINI_API_KEY=...`):
```bash
python run_all.py --input ./episodes --out outputs --limit 3       # try 3 episodes first
python run_all.py --input ./episodes --out outputs                 # all 50
```
Re-running is safe: every raw model reply is cached in `outputs/cache/`, so only failed or new calls are redone.
New episode? Drop the file in `./episodes` (or upload it in the app) and run again.

Outputs in `outputs/`: `promos_all.json` (everything considered, incl. rejected, with flags), `promos_ranked.csv`,
`post_first.md`, `failures.md` (auto-generated).

Then:
```bash
python cost_report.py --out outputs --shows-per-week 5     # tokens per episode / show / week (API-reported counts)
python evaluate.py --gold gold.json --out outputs          # hit rates vs YOUR best promos (see gold.example.json)
PROMO_PROVIDER_B=openai PROMO_MODEL_B=gpt-4o-mini python compare_models.py --out outputs   # 2 models, first 5 episodes
streamlit run app.py
```
`gold.json`: `[{"episode": 3, "start": "00:05.0", "end": "00:41.5"}, ...]` - write it yourself from watching episodes.
A hit = the tool's pick shares >= 50% of the shorter clip's screen time with a gold clip (`HIT_OVERLAP` in `config.py`).

## Deploy (Streamlit Community Cloud)

1. `git init`, commit, push to a **private** GitHub repo. Check `git status` first: no `episodes/`, `outputs/`, `.env`.
2. share.streamlit.io -> New app -> pick the repo, branch `main`, main file `app.py`.
3. App settings -> Secrets: paste the contents of `.streamlit/secrets.toml.example` with your real key and a password.
   `APP_PASSWORD` gates the app so strangers cannot spend your API quota; give the password to the reviewers.
4. Free-tier hosting is fine for a demo on 1-5 episodes. Run all 50 on your Mac (many API calls; sessions can time out).

## Limits (honest)

- Only the *code* parts are verified offline. Quality of the AI picks depends on the model; check `failures.md` and `evaluate.py`.
- Cross-episode promos only use the last/first 10 lines of neighbouring episodes as context.
- Code heuristics (hook/bridge candidates, "quiet start", "cue only") are rules of thumb, tuned on one show.
- Blind viewer runs on the top `PROMO_BLIND_K` (default 5) clips per episode only, to keep cost down.
- The tool picks moments from the screenplay text; it has not seen the video.
