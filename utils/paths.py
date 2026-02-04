"""Centralized path handling for lab scripts.

Use get_paths(__file__) in any script to get data_dir and output_dir
relative to that script's directory. Works regardless of current working directory.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def get_paths(script_file: str) -> SimpleNamespace:
    """Return script_dir, data_dir, and output_dir for the given script.

    Usage in a lab script:
        from utils.paths import get_paths
        paths = get_paths(__file__)
        df = pd.read_csv(paths.data_dir / "data.csv")
        paths.output_dir.mkdir(exist_ok=True)
        plt.savefig(paths.output_dir / "figure.png")
    """
    script_dir = Path(script_file).resolve().parent
    return SimpleNamespace(
        script_dir=script_dir,
        data_dir=script_dir / "data",
        output_dir=script_dir / "output",
    )
