# chord-transcriber

Automatic chord recognition from audio. Give it a song and it returns a timeline of chords (root, quality and seventh, e.g. `Am7`, `Fmaj7`, `G7`) plus a small HTML player that shows the chords in sync with the music.

It's a CNN on constant-Q spectrogram windows, followed by a second-order hidden semi-Markov model (HSMM) that decodes a chord sequence with realistic transitions and durations. Trained and evaluated on the RWC Popular Music Database.

## Results

15-song held-out test split from RWC Pop, macro-averaged over songs.

| | Root | Root + quality | Root + quality + 7th |
|---|---:|---:|---:|
| CNN only, continuous audio | 80.7% | 74.8% | 60.2% |
| **CNN + HSMM, continuous audio** | **83.3%** | **79.0%** | **65.9%** |
| CNN only, annotation-aligned windows | 89.9% | 84.4% | 71.1% |

The first two rows are scored on the same 0.2 s frames across the full songs, so they show what the HSMM adds. The last row runs the CNN on windows centred on the annotated chord segments, which is easier because it never sees chord boundaries. It's mostly useful as an upper bound for the acoustic model. Full per-song numbers are in [results/rwc_evaluation.json](results/rwc_evaluation.json).

## How it works

**Frontend.** The audio is resampled to 22.05 kHz and the harmonic part is separated out with HPSS. A CQT is computed over six octaves from C1 at 3 bins per semitone, then summed down to 72 semitone bins. The model sees 32-frame windows (about 0.75 s) every 0.2 s.

**CNN.** Rather than classifying 288 chords directly, the network factors a chord into parts:

1. A root encoder looks at the spectrum and predicts one of 12 roots.
2. For each of the top-k candidate roots, the spectrum is rotated so that root sits at pitch class 0, and a second encoder works on this root-relative view. This makes quality and seventh transposition-invariant, so a minor chord looks the same on A as on C#.
3. From the relative features: quality (maj, min, dim, aug, sus2, sus4), then seventh (none, b7, maj7, dim7) conditioned on the quality, then extensions (9ths, 11ths, 13ths) conditioned on both.

The heads combine into a posterior over 12 × 6 × 4 = 288 chord states. The extension head is trained as an auxiliary task but isn't part of the decoded state.

**HSMM.** Taking the argmax of each frame makes the chords flicker. The decoder runs a beam-pruned segmental Viterbi over the posteriors, where each segment's score combines:

- the acoustic likelihood from the CNN,
- a transition model over the previous *two* chords, counted in relative terms (interval from the previous root plus quality/seventh), so a ii–V–I is the same pattern in every key,
- a log-normal duration prior (segments between 0.4 s and 6 s).

## Usage

```bash
pip install -r requirements.txt
```

Drop a song (`.wav`, `.flac`, `.mp3` or `.ogg`) into `input/` and run:

```bash
python transcribe.py
```

For each song in `input/`, this writes `chords.csv`, `chords.json` and `viewer.html` to `output/<song>/`. Open the viewer in a browser to play the song with the current and upcoming chords. You can also pass files directly, e.g. `python transcribe.py ~/Music/song.mp3 --device mps`.

CNN posteriors are cached in `.cache/`, so re-running a song (e.g. with `--hsmm-order 1`) only redoes the decoding. If the song is one of the RWC test songs (e.g. `RWC_P039.flac`), the viewer also shows the ground truth from `data/annotations/`.

To reproduce the results table:

```bash
python scripts/evaluate.py
```

This uses the CQT windows and CNN posteriors saved in `data/`, so you don't need the RWC audio. [notebooks/demo.ipynb](notebooks/demo.ipynb) goes through the evaluation with plots, then transcribes whatever is in `input/`.

## Training

[scripts/train.py](scripts/train.py) trains the CNN from precomputed feature files (`RWC_PNNN.npz`, 72×32 windows with root/quality/seventh/extension labels) using the split in [models/split_manifest.json](models/split_manifest.json): 70 train, 15 validation, 15 test, seed 42. Each training window is transposed by a random number of semitones (rotating the pitch classes and the root label with it), and training stops early on validation root + quality + seventh accuracy. Only the test-split features are in this repo, so you'll need to generate the training features yourself to retrain.

## Layout

```
chordrec/        the package: frontend, CNN, HSMM, pipeline, viewer
transcribe.py    CLI entry point
input/           put songs here
scripts/         evaluation and training
models/          trained CNN checkpoint, transition counts, data split
data/            RWC test split: annotations, CQT windows, saved CNN posteriors
results/         evaluation output, including per-frame predictions for each song
notebooks/       walkthrough notebook
```

The RWC audio isn't included because it's licensed by AIST and can't be redistributed. If you have the database, put the test songs in `data/audio/` as `RWC_PNNN.wav` and `scripts/evaluate.py` will regenerate any missing posteriors from them.

## Limitations

- No bass or inversions: `C/E` is treated as `C`.
- Sixth chords are folded into major/minor, and extensions are predicted but not decoded.
- Training data is 70 songs of Japanese pop from one database, so it'll do worse on very different styles.
