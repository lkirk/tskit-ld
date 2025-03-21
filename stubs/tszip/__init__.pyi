from pathlib import Path

import tskit


def load(path: Path | int) -> tskit.TreeSequence: ...
def compress(
    ts: tskit.TreeSequence, destination: Path, variants_only: bool = False
) -> None: ...
