"""
Shot event logger and analytics.

Collects MergedShot events during the run and exports them at the end.
"""
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List

CSV_FIELDS = [
    "frame", "timestamp", "player_id",
    "shot_type", "side", "confidence", "source",
    "contact_x", "contact_y",
    "out_dx", "out_dy",
]

class EventLogger:
    def __init__(self, out_dir: str = "outputs"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events: list = []

    def add(self, ev) -> None:
        self.events.append(ev)

    def export(self,
               csv_path: str = "events.csv",
               json_path: str = "events.json",
               summary_path: str = "summary.json") -> dict:
        rows = [self._to_row(ev) for ev in self.events]

        # CSV
        with (self.out_dir / csv_path).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # JSON (events array)
        with (self.out_dir / json_path).open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        # Analytics summary
        summary = self._build_summary()
        with (self.out_dir / summary_path).open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    # ------------------------------------------------------------------

    def _to_row(self, ev) -> dict:
        cxy = getattr(ev, "contact_xy", None) or (None, None)
        odir = getattr(ev, "out_dir", None) or (None, None)
        return {
            "frame":      ev.frame,
            "timestamp":  round(ev.timestamp, 3),
            "player_id":  ev.player_id,
            "shot_type":  ev.shot_type,
            "side":       ev.side,
            "confidence": ev.confidence,
            "source":     ev.source,
            "contact_x":  None if cxy[0] is None else round(cxy[0], 1),
            "contact_y":  None if cxy[1] is None else round(cxy[1], 1),
            "out_dx":     None if odir[0] is None else round(odir[0], 3),
            "out_dy":     None if odir[1] is None else round(odir[1], 3),
        }

    def _build_summary(self) -> dict:
        by_type    = Counter(e.shot_type  for e in self.events)
        by_conf    = Counter(e.confidence for e in self.events)
        by_source  = Counter(e.source     for e in self.events)
        per_player = defaultdict(lambda: Counter())
        for e in self.events:
            per_player[e.player_id][e.shot_type] += 1

        # high/med-only counts (the "trust-it" view)
        trusted   = [e for e in self.events if e.confidence in ("high", "med")]
        trusted_by_type = Counter(e.shot_type for e in trusted)

        return {
            "total_events":          len(self.events),
            "trusted_events":        len(trusted),
            "by_shot_type":          dict(by_type),
            "by_confidence":         dict(by_conf),
            "by_source":             dict(by_source),
            "trusted_by_shot_type":  dict(trusted_by_type),
            "per_player": {
                str(pid): dict(c) for pid, c in per_player.items()
            },
        }


def player_counts(events: list) -> dict:
    """Live per-player Counter — used by main.py to draw on-video overlay."""
    pp = defaultdict(lambda: Counter())
    for e in events:
        pp[e.player_id][e.shot_type] += 1
    return pp
