"""Chord sequences as (start, end, label) intervals."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChordTimeline:
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    labels: NDArray[np.str_]

    def write_csv(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("start_seconds", "end_seconds", "chord"))
            writer.writerows(
                (f"{start:.6f}", f"{end:.6f}", label)
                for start, end, label in zip(self.starts, self.ends, self.labels)
            )

    def write_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        events = [
            {
                "start_seconds": float(start),
                "end_seconds": float(end),
                "chord": str(label),
            }
            for start, end, label in zip(self.starts, self.ends, self.labels)
        ]
        target.write_text(json.dumps(events, indent=2), encoding="utf-8")
