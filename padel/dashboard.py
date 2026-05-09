import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(summary_path: str = "outputs/summary.json",
           out_png: str = "outputs/dashboard.png") -> str:
    sp = Path(summary_path)
    if not sp.exists():
        raise FileNotFoundError(f"summary.json not found at {sp}")

    with sp.open() as f:
        s = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Padel Match Analytics", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    bt = s.get("by_shot_type", {})
    if bt:
        ax.bar(bt.keys(), bt.values(), color=["#4e79a7", "#f28e2b", "#e15759"])
    ax.set_title(f"Shots by Type  (total={s.get('total_events', 0)})")
    ax.set_ylabel("count")

    ax = axes[0, 1]
    pp = s.get("per_player", {})
    types = sorted({t for c in pp.values() for t in c})
    bottoms = [0] * len(pp)
    for t in types:
        vals = [pp[p].get(t, 0) for p in pp]
        ax.bar(list(pp.keys()), vals, bottom=bottoms, label=t)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title("Per-player shot counts")
    ax.set_xlabel("player_id")
    if types:
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    bc = s.get("by_confidence", {})
    order = ["high", "med", "low"]
    keys  = [k for k in order if k in bc]
    vals  = [bc[k] for k in keys]
    if keys:
        ax.bar(keys, vals, color=["#59a14f", "#edc948", "#bab0ac"])
    ax.set_title(f"Events by Confidence  (trusted={s.get('trusted_events', 0)})")
    ax.set_ylabel("count")

    ax = axes[1, 1]
    bd = s.get("by_direction", {})
    if bd:
        items = sorted(bd.items(), key=lambda kv: -kv[1])
        labels = [k for k, _ in items]
        vals   = [v for _, v in items]
        ax.barh(labels, vals, color="#76b7b2")
        ax.invert_yaxis()
    ax.set_title(f"Shot direction  |  bounces detected: {s.get('total_bounces', 0)}")
    ax.set_xlabel("count")

    plt.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[dashboard] wrote {out}")
    return str(out)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "outputs/summary.json"
    render(p)
