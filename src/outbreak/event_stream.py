"""
Build a time-ordered event stream from agent trajectory histories (Handoff 9).

The static contact graph (colocation_counter_capped → build_contact_edges)
aggregates co-locations into an atemporal graph.  This module exposes the
raw time-resolved stream so the outbreak engine can consume it without
collapsing.  The two representations are parallel; neither replaces the other.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def build_event_stream(
    trajectories,
    space_bin: float = 0.5,
    time_bin: int = 5,
) -> List[Tuple[int, Tuple[int, int], List[int]]]:
    """
    Convert trajectory visit histories to a sorted event stream.

    Returns
    -------
    List of (t_bin, (sx, sy), [agent_ids]) sorted by (t_bin, sx, sy).
    Each entry is one venue-time event: agents co-present in a
    space_bin × space_bin × time_bin cell.  Only events with ≥2 agents
    are returned (no exposure otherwise).

    Parameters
    ----------
    trajectories : list of Trajectory (from generate_walks)
    space_bin    : spatial bin width (km); must match the generator value
    time_bin     : steps per time window; must match the generator value
    """
    bin_map: Dict[Tuple[int, int, int], List[int]] = {}
    for traj in trajectories:
        for x, y, step in traj.visits:
            sx = int(np.floor(x / space_bin))
            sy = int(np.floor(y / space_bin))
            tb = step // time_bin
            key = (tb, sx, sy)
            if key not in bin_map:
                bin_map[key] = []
            bin_map[key].append(traj.agent_id)

    events: List[Tuple[int, Tuple[int, int], List[int]]] = []
    for (tb, sx, sy), agents in bin_map.items():
        unique = list(set(agents))
        if len(unique) >= 2:
            events.append((tb, (sx, sy), unique))

    events.sort(key=lambda e: (e[0], e[1]))
    return events


def tile_event_stream(
    stream: List[Tuple[int, Tuple[int, int], List[int]]],
    n_tiles: int,
    n_bins_per_tile: int,
) -> List[Tuple[int, Tuple[int, int], List[int]]]:
    """
    Tile a base event stream n_tiles times to extend epidemic horizon.

    Each tile repeats the same venue visit pattern, shifting t_bin by
    tile_index × n_bins_per_tile.  Mechanistically: PWID return to the
    same venues repeatedly — the 80-step trajectory captures the habitual
    pattern; tiling extends it over months without re-running the generator.

    Parameters
    ----------
    stream         : base event stream from build_event_stream()
    n_tiles        : number of repetitions (e.g., 10 → 10 × 16 bins = 800 days)
    n_bins_per_tile : n_steps // time_bin for the base trajectory (e.g., 80//5=16)
    """
    tiled: List[Tuple[int, Tuple[int, int], List[int]]] = []
    for tile in range(n_tiles):
        shift = tile * n_bins_per_tile
        for t_bin, cell, agents in stream:
            tiled.append((t_bin + shift, cell, agents))
    return tiled


def occupancy_stats(
    event_stream: List[Tuple[int, Tuple[int, int], List[int]]],
) -> Dict:
    """Summary statistics of per-event occupancy (diagnostic)."""
    occupancies = [len(agents) for _, _, agents in event_stream]
    if not occupancies:
        return {}
    arr = np.array(occupancies)
    hist = dict(Counter(occupancies))
    return {
        "n_events": len(occupancies),
        "mean_occupancy": float(np.mean(arr)),
        "max_occupancy": int(np.max(arr)),
        "p90_occupancy": float(np.percentile(arr, 90)),
        "histogram": {str(k): v for k, v in sorted(hist.items())},
    }
