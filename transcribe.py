"""Transcribe the chords of one or more audio files.

    python transcribe.py                 # everything in input/
    python transcribe.py song.mp3 another.wav

For each song this writes chords.csv, chords.json and viewer.html to
<output-dir>/<song>/. If a matching annotation file exists in
--annotation-dir, the viewer shows it next to the prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chordrec import (
    ChordPipeline,
    ChordTimeline,
    ModelArtifacts,
    load_annotations,
    write_interactive_viewer,
)

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "audio", type=Path, nargs="*", help="Audio file(s); defaults to everything in --input-dir."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/chordrec"))
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: auto).")
    parser.add_argument("--top-roots", type=int, default=4, choices=range(1, 13))
    parser.add_argument("--hsmm-order", type=int, default=2, choices=(1, 2))
    return parser


def reference_timeline(path: Path, duration: float) -> ChordTimeline | None:
    starts, ends, labels = load_annotations(path).intervals(duration)
    return ChordTimeline(starts, ends, labels) if len(labels) else None


def main() -> int:
    args = build_parser().parse_args()
    if not args.audio:
        args.audio = sorted(
            path for path in args.input_dir.glob("*") if path.suffix.lower() in AUDIO_SUFFIXES
        )
        if not args.audio:
            print(f"No audio found in {args.input_dir}/. Drop a song there or pass a path.")
            return 1
    for path in args.audio:
        if not path.is_file():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 1
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            print(f"error: unsupported audio format: {path}", file=sys.stderr)
            return 1

    pipeline = ChordPipeline(
        ModelArtifacts.from_dir(args.models_dir),
        device=args.device,
        top_roots=args.top_roots,
        hsmm_order=args.hsmm_order,
        cache_dir=args.cache_dir,
    )
    for path in args.audio:
        print(f"{path.name}: running CNN + HSMM ...")
        timeline, _ = pipeline.process(path)
        folder = args.output_dir / path.stem
        timeline.write_csv(folder / "chords.csv")
        timeline.write_json(folder / "chords.json")

        annotation = args.annotation_dir / f"{path.stem}.csv"
        reference = None
        if annotation.is_file():
            reference = reference_timeline(annotation, float(timeline.ends[-1]))
        viewer = write_interactive_viewer(
            path.stem,
            path,
            timeline,
            folder / "viewer.html",
            model_name="Hierarchical CNN + second-order HSMM",
            reference=reference,
        )
        print(f"  {len(timeline.labels)} chord segments -> {folder}/")
        print(f"  open {viewer} to play it back with the chords")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
