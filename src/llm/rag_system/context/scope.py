from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GenomeScope:
    """Immutable genome context propagated across toolcalls.

    Fields:
      genome_id: canonical genome identifier
      contig_ids: tuple of contig ids (may be empty)
      coordinate_window: (start, end) tuple in bp (may be (0, 0) if unspecified)
    """

    genome_id: str
    contig_ids: Tuple[str, ...]
    coordinate_window: Tuple[int, int]

