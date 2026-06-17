"""
Task 3: Candidate attractor-layer proxies for the mobility network generator.

Three clearly-labeled candidate configurations for AC to choose from.
No proxy is enshrined as final. The drug-market choice is AC's judgment call.

CANDIDATE A — SSP-only (defensible baseline):
  Only SSP/SEP venues as attractors. The only real, source-able attractor layer.
  Assumption: PWID activity clusters around harm-reduction infrastructure.
  Drug markets: NOT modeled (no approved proxy).

CANDIDATE B — SSP + diffuse market field (moderate assumption):
  SSP venues + synthetic drug-market venues spread across activity space.
  Assumption: drug markets are distributed (no single hotspot).
  Assumption explicitly stated; data source: PLACEHOLDER.

CANDIDATE C — SSP + concentrated hotspot clusters (strong assumption):
  SSP venues + 2 tightly-concentrated market/hotspot clusters.
  Assumption: a small number of fixed sites dominate drug market activity.
  Models the "shooting gallery" or known open-air market concentration.
  Assumption explicitly stated; data source: PLACEHOLDER.

ETHICAL INVARIANT: All proxies generate ensembles from population distributions.
No individual trajectories. Drug-market proxies are flagged as assumptions pending AC.
"""

from __future__ import annotations

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mobility.network_generator import Venue

PROXY_DESCRIPTIONS = {
    "ssp_only": (
        "CANDIDATE A — SSP-only: Only SSP/SEP venues as attractors. "
        "The only defensible, source-able attractor layer (NASEN-style). "
        "No drug market proxy. Represents minimum venue-anchoring assumption."
    ),
    "ssp_diffuse_market": (
        "CANDIDATE B — SSP + diffuse market field: SSP venues plus synthetic "
        "drug market venues spread across the activity space at moderate density. "
        "ASSUMPTION: markets are distributed (no single hotspot). "
        "SOURCE: PLACEHOLDER — proxy only. AC must approve before use in results."
    ),
    "ssp_concentrated_hotspot": (
        "CANDIDATE C — SSP + concentrated hotspot clusters: SSP venues plus "
        "2 tightly-clustered market/hotspot concentrations. "
        "ASSUMPTION: a small number of fixed high-activity sites dominate. "
        "Models 'shooting gallery' or known open-air market concentration. "
        "SOURCE: PLACEHOLDER — proxy only. AC must approve before use in results."
    ),
}


def make_ssp_venues(
    rg_scale_km: float,
    n_ssp: int = 8,
    rng: np.random.Generator = None,
) -> list:
    """
    Generate synthetic SSP/SEP venue layer.

    Venues are placed in a spatial grid to ensure coverage across the activity
    space rather than random clustering (avoids accidental degeneracy).
    Relevance = 2.0 (high — real anchors).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    venues = []
    # Place SSPs in a rough grid across the activity space ± rg_scale_km
    n_side = max(2, int(np.ceil(np.sqrt(n_ssp))))
    coords = np.linspace(-rg_scale_km * 0.8, rg_scale_km * 0.8, n_side)
    idx = 0
    for xi in coords:
        for yi in coords:
            if idx >= n_ssp:
                break
            # Small random jitter so venues aren't perfectly on-grid
            jx = rng.uniform(-rg_scale_km * 0.1, rg_scale_km * 0.1)
            jy = rng.uniform(-rg_scale_km * 0.1, rg_scale_km * 0.1)
            venues.append(Venue(
                venue_id=idx,
                x=float(xi + jx),
                y=float(yi + jy),
                relevance=2.0,
                venue_type="ssp",
            ))
            idx += 1
    return venues


def make_proxy_a_ssp_only(
    rg_scale_km: float = 2.4,
    n_ssp: int = 8,
    rng: np.random.Generator = None,
) -> list:
    """Candidate A: SSP venues only."""
    if rng is None:
        rng = np.random.default_rng(42)
    return make_ssp_venues(rg_scale_km, n_ssp, rng)


def make_proxy_b_ssp_diffuse_market(
    rg_scale_km: float = 2.4,
    n_ssp: int = 8,
    n_market: int = 6,
    market_relevance: float = 1.2,
    rng: np.random.Generator = None,
) -> list:
    """
    Candidate B: SSP + diffuse drug-market venues.

    Market venues are spread across the activity space at moderate density.
    relevance = 1.2 (lower than SSP = 2.0; assumption: SSPs are stronger anchors).

    PLACEHOLDER — market locations are synthetic. AC must approve a real proxy.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    venues = make_ssp_venues(rg_scale_km, n_ssp, rng)

    # Markets scattered across activity space
    mx = rng.uniform(-rg_scale_km, rg_scale_km, n_market)
    my = rng.uniform(-rg_scale_km, rg_scale_km, n_market)
    for i, (x, y) in enumerate(zip(mx, my)):
        venues.append(Venue(
            venue_id=n_ssp + i,
            x=float(x),
            y=float(y),
            relevance=float(market_relevance),
            venue_type="drug_market_diffuse_PLACEHOLDER",
        ))
    return venues


def make_proxy_c_ssp_concentrated_hotspot(
    rg_scale_km: float = 2.4,
    n_ssp: int = 8,
    hotspot_relevance: float = 3.0,
    rng: np.random.Generator = None,
) -> list:
    """
    Candidate C: SSP + 2 concentrated hotspot clusters.

    2 high-relevance hotspot nodes model shooting-gallery or open-air-market
    concentration. Agents will cluster strongly at these points, creating a
    heavier-tailed degree distribution and higher clustering.

    hotspot_relevance = 3.0 (higher than SSP = 2.0; the dominant attractor).

    PLACEHOLDER — hotspot locations are synthetic. AC must approve a real proxy.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    venues = make_ssp_venues(rg_scale_km, n_ssp, rng)

    # 2 concentrated hotspot clusters: one near-center, one at periphery
    hotspot_centers = [
        (rg_scale_km * 0.2, rg_scale_km * 0.1),   # near center
        (-rg_scale_km * 0.6, rg_scale_km * 0.5),  # peripheral
    ]
    cluster_scatter = rg_scale_km * 0.05  # very tight — ~50m if rg=1km

    vid = n_ssp
    for cx, cy in hotspot_centers:
        # 3 closely-spaced sub-venues per hotspot (realistic for a venue cluster)
        for _ in range(3):
            x = cx + rng.normal(0, cluster_scatter)
            y = cy + rng.normal(0, cluster_scatter)
            venues.append(Venue(
                venue_id=vid,
                x=float(x),
                y=float(y),
                relevance=float(hotspot_relevance),
                venue_type="drug_market_hotspot_PLACEHOLDER",
            ))
            vid += 1
    return venues


def all_proxies(rg_scale_km: float = 2.4, rng: np.random.Generator = None) -> dict:
    """Return all 3 candidate venue layers keyed by proxy name."""
    if rng is None:
        rng = np.random.default_rng(42)
    # Use separate sub-RNGs so proxies don't interfere
    return {
        "ssp_only": make_proxy_a_ssp_only(rg_scale_km, rng=np.random.default_rng(42)),
        "ssp_diffuse_market": make_proxy_b_ssp_diffuse_market(rg_scale_km, rng=np.random.default_rng(43)),
        "ssp_concentrated_hotspot": make_proxy_c_ssp_concentrated_hotspot(rg_scale_km, rng=np.random.default_rng(44)),
    }
