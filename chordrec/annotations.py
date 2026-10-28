"""Read RWC chord annotations and map them onto the model's vocabulary."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def normalize_chord(label: object) -> str:
    """Normalize an annotation label to the model's chord names.

    Bass inversions are dropped because the model doesn't predict bass notes,
    and sixth chords fall back to their major/minor quality.
    """

    chord = str(label).strip().strip("'\"")
    if chord.lower() in {"", "n", "none", "x", "?"}:
        return "N"
    chord = chord.split("/", 1)[0].replace("_", "")
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(?::)?(.+)?", chord)
    if not match:
        return chord.replace(":", "")

    root = match.group(1).upper() + match.group(2)
    descriptor = (match.group(3) or "maj").lower()
    if descriptor == "m" or descriptor.startswith("min"):
        quality = "m"
    elif descriptor.startswith("dim"):
        quality = "dim"
    elif descriptor.startswith("aug"):
        quality = "aug"
    elif descriptor.startswith("sus2"):
        quality = "sus2"
    elif descriptor.startswith("sus4") or descriptor.startswith("sus"):
        quality = "sus4"
    else:
        quality = ""

    if "maj7" in descriptor or (
        descriptor.startswith("maj") and descriptor.endswith(("9", "11", "13"))
    ):
        seventh = "maj7"
    elif "dim7" in descriptor:
        seventh = "dim7"
    elif "7" in descriptor or descriptor.endswith(("9", "11", "13")):
        seventh = "7"
    else:
        seventh = ""

    # The saved formatter represents diminished quality and seventh separately.
    if quality == "dim" and seventh == "dim7":
        return root + "dimdim7"
    return root + quality + seventh


@dataclass(frozen=True)
class ChordAnnotations:
    """Chord labels placed at annotated beat times."""

    times: NDArray[np.float64]
    labels: NDArray[np.str_]

    def intervals(
        self, audio_duration: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.str_]]:
        """Convert beat timestamps to valid intervals within the audio."""

        if audio_duration <= 0:
            raise ValueError("Audio duration must be positive.")
        if len(self.times) == 0:
            return self.times.copy(), self.times.copy(), self.labels.copy()
        typical = float(np.median(np.diff(self.times))) if len(self.times) > 1 else 1.0
        ends = np.minimum(
            np.r_[self.times[1:], self.times[-1] + typical], audio_duration
        )
        valid = (self.times >= 0) & (self.times < audio_duration) & (ends > self.times)
        return self.times[valid], ends[valid], self.labels[valid]


def load_annotations(path: Path | str) -> ChordAnnotations:
    """Load a ``t_start;chord`` CSV and merge duplicate timestamps.

    If two rows at the same time disagree, that position becomes ``N``.
    """

    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            try:
                rows.append((float(row["t_start"]), row["chord"]))
            except (KeyError, TypeError, ValueError):
                continue
    by_time: dict[float, list[str]] = {}
    for timestamp, label in rows:
        by_time.setdefault(round(timestamp, 6), []).append(normalize_chord(label))
    cleaned = [
        (timestamp, labels[0] if len(set(labels)) == 1 else "N")
        for timestamp, labels in sorted(by_time.items())
    ]
    return ChordAnnotations(
        times=np.asarray([row[0] for row in cleaned], dtype=np.float64),
        labels=np.asarray([row[1] for row in cleaned], dtype="<U24"),
    )
