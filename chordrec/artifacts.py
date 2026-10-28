"""Paths to the trained CNN, the harmonic language model, and the data split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifacts:
    cnn_checkpoint: Path
    language_model: Path
    split_manifest: Path

    @classmethod
    def from_dir(cls, directory: Path | str = "models") -> "ModelArtifacts":
        root = Path(directory).expanduser()
        return cls(
            root / "best_model.pt",
            root / "harmonic_model.pkl",
            root / "split_manifest.json",
        )

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        return self.cnn_checkpoint, self.language_model, self.split_manifest

    def validate(self) -> "ModelArtifacts":
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing model file(s): " + ", ".join(map(str, missing))
            )
        return self
