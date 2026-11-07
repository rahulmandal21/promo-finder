"""Pick the 5 promos to post first to a cold audience, and list strong clips deliberately held back."""
import config
from ranker import base_score, needs_earlier
from utils import episodes_of, home_episode, segs_str


def _early_bonus(p):
    """Up to +2.0 for episode 1, fading to 0 by episode 16: a cold viewer should meet the show near its start."""
    return 2.0 * max(0.0, 1.0 - (home_episode(p) - 1) / 15)


def choose(picks, all_scored):
    eligible = [p for p in picks if not needs_earlier(p)]
    for p in eligible:
        p["post_first_score"] = round(p["rank_score"] + _early_bonus(p) + (0.5 if p.get("blind", {}).get("ok") else 0), 3)
    eligible.sort(key=lambda x: -x["post_first_score"])
    chosen, used_eps = [], set()
    for p in eligible:
        if home_episode(p) in used_eps:
            continue
        chosen.append(p)
        used_eps.add(home_episode(p))
        if len(chosen) == config.POST_FIRST_N:
            break
    skipped = sorted((p for p in all_scored if p.get("scores") and base_score(p) >= config.SKIP_MIN_BASE
                      and needs_earlier(p)), key=lambda x: -base_score(x))[:10]
    return chosen, skipped


def _why(p):
    b = p.get("blind")
    cold = "a blind cold-viewer test found it understandable on its own" if b and b["ok"] else \
        "it scored well on standalone clarity"
    early = ("an early episode, so little backstory is needed" if home_episode(p) <= 10
             else "a later episode, picked only because it still works cold")
    return (f"Episode {home_episode(p)} ({p['shape']}, score {p['rank_score']:.1f}): {cold}; {early}. "
            f"Hook: \"{p['hook_line'][:90]}\"")


def to_markdown(chosen, skipped, provider):
    out = ["# Post these first (cold audience)\n"]
    if provider == "mock":
        out.append("> **MOCK RUN** - scores come from a test heuristic, not a real model. Do not use for decisions.\n")
    if not chosen:
        out.append("No promo cleared the bar for a cold audience.\n")
    for i, p in enumerate(chosen, 1):
        out.append(f"{i}. **{segs_str(p)}** - {_why(p)}")
    out.append("\n## High-scoring clips deliberately skipped (they need earlier context)\n")
    if not skipped:
        out.append("None above the threshold.")
    for p in skipped:
        why = p["blind"]["confusion"] if p.get("blind") and p["blind"].get("confusion") else "low standalone score"
        out.append(f"- **{segs_str(p)}** (base {base_score(p):.1f}, {p['source']}): {why or 'low standalone score'}")
    return "\n".join(out) + "\n"
