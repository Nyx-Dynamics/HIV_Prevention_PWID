"""
Mobility-driven contact-network generator for PWID epidemic threshold estimation.

Pipeline (5 pure functions, each tested independently):
  1. sample_traversement_potential   → per-agent radius-of-gyration r_g
  2. generate_walks                  → agent trajectory histories (EPR)
  3. colocations                     → proximity opportunities (co-location pairs)
  4. sharing_edges                   → Bernoulli sharing decisions
  5. build_contact_graph             → NetworkX graph + degree stats

ETHICAL INVARIANT: This module generates an ensemble sampled from population
distributions. It must never track, reconstruct, or require real individuals'
trajectories. A digital twin of PWID would be surveillance of a criminalized
population and is explicitly out of scope. Do not upgrade it toward one.

Walk engine: pure-NumPy EPR (Exploration + Preferential Return).
scikit-mobility DensityEPR would be a drop-in upgrade if geo deps are available.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    """A synthetic agent drawn from population distributions. NOT a real person."""
    agent_id: int
    rg: float           # radius of gyration (km)
    hiv_positive: bool  # seeded from prevalence distribution


@dataclass
class Venue:
    """An attractor point (SSP, drug market proxy, etc.)."""
    venue_id: int
    x: float            # location coordinate
    y: float
    relevance: float    # attractor weight (≥0)
    venue_type: str     # "ssp" | "drug_market_placeholder" | "generic"


@dataclass
class Trajectory:
    """Location visit history for one agent."""
    agent_id: int
    # List of (x, y, step_index) tuples
    visits: List[Tuple[float, float, int]] = field(default_factory=list)

    def location_counts(self) -> Dict[Tuple[float, float], int]:
        """Visit frequency per (x, y) bin."""
        counts: Dict[Tuple[float, float], int] = {}
        for x, y, _ in self.visits:
            counts[(x, y)] = counts.get((x, y), 0) + 1
        return counts


@dataclass
class NetworkStats:
    """Degree-distribution statistics from the generated contact graph."""
    n_nodes: int
    n_edges: int
    mean_degree: float
    k2_moment: float        # ⟨k²⟩ — most sensitive to tail; treat as range
    degree_sequence: List[int]
    giant_component_fraction: float
    mean_path_length: Optional[float]   # None if disconnected
    clustering_coefficient: float


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Traversement potential
# ─────────────────────────────────────────────────────────────────────────────

def sample_traversement_potential(
    n: int,
    rg_scale_km: float,
    rg_growth_exponent: float,
    seed_hiv_prevalence: float,
    rng: np.random.Generator,
) -> List[Agent]:
    """
    Draw n synthetic agents. Each agent gets a radius-of-gyration r_g sampled
    from a heavy-tailed distribution scaled to the PWID activity radius.

    r_g follows a truncated power-law consistent with the empirical exponent ζ:
      P(r_g) ∝ r_g^{−ζ}   for r_g ∈ [r_min, r_max]

    HIV seeding is drawn from seed_hiv_prevalence (Bernoulli).

    Parameters
    ----------
    n : number of agents
    rg_scale_km : mean activity radius (km) — from PWID survey
    rg_growth_exponent : ζ — radius-of-gyration growth exponent
    seed_hiv_prevalence : fraction of agents initialized HIV+
    rng : seeded numpy Generator
    """
    r_min = rg_scale_km * 0.1
    r_max = rg_scale_km * 10.0

    # Truncated power-law via inverse CDF
    alpha = rg_growth_exponent  # shape parameter
    u = rng.uniform(size=n)
    # Inverse CDF for P(r) ∝ r^{-α}, α > 1
    if alpha != 1.0:
        r_g_vals = (r_min**(1 - alpha) + u * (r_max**(1 - alpha) - r_min**(1 - alpha))) ** (1 / (1 - alpha))
    else:
        r_g_vals = r_min * np.exp(u * np.log(r_max / r_min))
    r_g_vals = np.clip(r_g_vals, r_min, r_max)

    hiv_flags = rng.random(n) < seed_hiv_prevalence

    return [Agent(agent_id=i, rg=float(r_g_vals[i]), hiv_positive=bool(hiv_flags[i]))
            for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EPR walks (pure-NumPy fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _truncated_powerlaw_jump(beta: float, r_scale: float, rng: np.random.Generator) -> float:
    """Sample jump distance from truncated power-law P(Δr) ∝ (Δr)^{−1−β}."""
    r_min, r_max = 0.01 * r_scale, 5.0 * r_scale
    u = rng.uniform()
    if beta != 0:
        return (r_min**(-beta) + u * (r_max**(-beta) - r_min**(-beta))) ** (-1 / beta)
    return r_min * (r_max / r_min) ** u


def _attractor_pull(
    current_x: float, current_y: float,
    venues: List[Venue],
    rg: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Return a jump destination biased toward nearby high-relevance venues.
    Falls back to uniform random jump if no venues defined.
    """
    if not venues:
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.exponential(rg * 0.5)
        return current_x + dist * np.cos(angle), current_y + dist * np.sin(angle)

    vx = np.array([v.x for v in venues])
    vy = np.array([v.y for v in venues])
    vr = np.array([v.relevance for v in venues])

    dists = np.sqrt((vx - current_x) ** 2 + (vy - current_y) ** 2)
    # Gravity: relevance / (distance + small_constant)
    weights = vr / (dists + 0.01)
    weights /= weights.sum()
    idx = rng.choice(len(venues), p=weights)
    # Land near the chosen venue with Gaussian scatter
    scatter = rg * 0.1
    return float(venues[idx].x + rng.normal(0, scatter)), \
           float(venues[idx].y + rng.normal(0, scatter))


def generate_walks(
    agents: List[Agent],
    venues: List[Venue],
    n_steps: int,
    epr_rho: float,
    epr_gamma: float,
    jump_length_exponent: float,
    rng: np.random.Generator,
) -> List[Trajectory]:
    """
    EPR walk for each agent over n_steps.

    At each step:
      - Exploration with probability p_explore = ρ · S^{-γ}
        → attractor-biased truncated power-law jump
      - Preferential return ∝ visit frequency otherwise

    Parameters
    ----------
    agents : list of Agent objects
    venues : attractor layer (SSPs, drug markets, etc.)
    n_steps : number of mobility steps per simulation period
    epr_rho, epr_gamma : EPR exploration parameters
    jump_length_exponent : β for truncated power-law jump distance
    rng : seeded Generator
    """
    trajectories = []

    for agent in agents:
        # Initial position: place near a random venue or at origin
        if venues:
            start = rng.choice(len(venues))
            x = venues[start].x + rng.normal(0, 0.1)
            y = venues[start].y + rng.normal(0, 0.1)
        else:
            x, y = rng.normal(0, agent.rg), rng.normal(0, agent.rg)

        visits = [(float(x), float(y), 0)]
        location_counts: Dict[Tuple[float, float], int] = {(float(x), float(y)): 1}

        for step in range(1, n_steps):
            n_distinct = len(location_counts)  # S

            p_explore = min(epr_rho * (n_distinct ** (-epr_gamma)), 1.0)

            if rng.random() < p_explore:
                # Explore: jump to new location, biased toward attractors
                dx, dy = _attractor_pull(x, y, venues, agent.rg, rng)
                x, y = dx, dy
            else:
                # Preferential return: return to a previously visited location
                locs = list(location_counts.keys())
                counts = np.array([location_counts[l] for l in locs], dtype=float)
                counts /= counts.sum()
                idx = rng.choice(len(locs), p=counts)
                x, y = locs[idx]

            loc_key = (float(x), float(y))
            location_counts[loc_key] = location_counts.get(loc_key, 0) + 1
            visits.append((float(x), float(y), step))

        trajectories.append(Trajectory(agent_id=agent.agent_id, visits=visits))

    return trajectories


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Co-locations
# ─────────────────────────────────────────────────────────────────────────────

def colocations(
    trajectories: List[Trajectory],
    space_bin: float = 0.1,
    time_bin: int = 5,
) -> List[Tuple[int, int]]:
    """
    Find pairs of agents who co-locate (same spatial bin, same time bin).

    Each co-location is a *proximity opportunity* — NOT an edge.
    Two stages are kept distinct in code (co-location ≠ sharing).

    Parameters
    ----------
    trajectories : output of generate_walks
    space_bin : spatial bin size (same units as x, y coordinates, default km)
    time_bin : number of steps that define a temporal bin

    MODELING CHOICES: space_bin and time_bin are free parameters —
    they are exposed as arguments (not hardcoded) and flagged here
    as sensitivity axes for AC.
    """
    # Index visits by (spatial_bin, temporal_bin) → [(agent_id, ...)]
    bin_map: Dict[Tuple[int, int, int], List[int]] = {}

    for traj in trajectories:
        for x, y, step in traj.visits:
            sx = int(np.floor(x / space_bin))
            sy = int(np.floor(y / space_bin))
            tb = step // time_bin
            key = (sx, sy, tb)
            if key not in bin_map:
                bin_map[key] = []
            bin_map[key].append(traj.agent_id)

    # All unordered pairs in the same bin
    coloc_pairs: List[Tuple[int, int]] = []
    for agents_in_bin in bin_map.values():
        unique = list(set(agents_in_bin))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                coloc_pairs.append((unique[i], unique[j]))

    return coloc_pairs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Sharing edges (two-stage: co-location → Bernoulli sharing)
# ─────────────────────────────────────────────────────────────────────────────

def sharing_edges(
    coloc_pairs: List[Tuple[int, int]],
    sharing_prob: float,
    kappa_share: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Convert co-location opportunities into sharing edges via Bernoulli draw.

    Per-event sharing probability = sharing_prob * kappa_share.

    sharing_prob : NHBS 12-month prevalence (syringe or equipment sharing).
                  UNIT MISMATCH: this is 12-mo per-person, not per-event.
                  See params.py for the kappa_share calibration TODO.
    kappa_share : calibration scalar (per-event rate / 12-mo prevalence).
                  Default = 1.0 (conservative ceiling). Must be calibrated.
    rng : seeded Generator

    Returns unique edges (de-duplicated; self-loops excluded).
    """
    p_edge = min(sharing_prob * kappa_share, 1.0)

    edges = set()
    for a, b in coloc_pairs:
        if a != b and rng.random() < p_edge:
            edges.add((min(a, b), max(a, b)))  # canonical order

    return list(edges)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Contact graph
# ─────────────────────────────────────────────────────────────────────────────

def build_contact_graph(
    edges: List[Tuple[int, int]],
    n_agents: int,
) -> Tuple[nx.Graph, NetworkStats]:
    """
    Build an undirected contact graph from edge list and compute degree statistics.

    Parameters
    ----------
    edges : list of (agent_i, agent_j) sharing edges
    n_agents : total number of agents (for isolated nodes)

    Returns
    -------
    (graph, stats) where stats contains ⟨k⟩, ⟨k²⟩, giant-component, path length.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_agents))
    G.add_edges_from(edges)

    degree_seq = [d for _, d in G.degree()]
    k_arr = np.array(degree_seq, dtype=float)

    mean_k = float(np.mean(k_arr))
    k2 = float(np.mean(k_arr ** 2))

    # Giant component
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    giant_frac = len(components[0]) / n_agents if components else 0.0

    # Mean shortest path (only within giant component; None if size < 2)
    giant_sub = G.subgraph(components[0]) if components else G
    if len(giant_sub) >= 2:
        try:
            mpl = nx.average_shortest_path_length(giant_sub)
        except nx.NetworkXError:
            mpl = None
    else:
        mpl = None

    cc = float(nx.average_clustering(G))

    stats = NetworkStats(
        n_nodes=n_agents,
        n_edges=len(edges),
        mean_degree=mean_k,
        k2_moment=k2,
        degree_sequence=degree_seq,
        giant_component_fraction=giant_frac,
        mean_path_length=mpl,
        clustering_coefficient=cc,
    )

    return G, stats


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_generator(
    n_agents: int = 300,
    n_steps: int = 50,
    rg_scale_km: float = 2.4,
    rg_growth_exponent: float = 1.65,
    seed_hiv_prevalence: float = 0.07,
    epr_rho: float = 0.60,
    epr_gamma: float = 0.21,
    jump_length_exponent: float = 0.60,
    sharing_prob: float = 0.27,
    kappa_share: float = 0.01,
    space_bin: float = 0.5,
    time_bin: int = 5,
    venues: Optional[List[Venue]] = None,
    seed: int = 42,
) -> Tuple[nx.Graph, NetworkStats, List[Agent]]:
    """
    Run the full 5-step pipeline with a single call.

    All parameters have documented defaults (sourced or PLACEHOLDER).
    space_bin and time_bin are modeling choices — expose them; flag for AC.

    kappa_share default = 0.01 is an approximate calibration producing ⟨k⟩ ≈ 2–4
    for typical inputs. The conservative ceiling (1.0) is documented in params.py.
    The correct value requires: daily injection frequency × fraction shared × mean
    co-location rate — see SOURCING_mobility_network.md for the TODO.

    Returns
    -------
    (graph, stats, agents)
    """
    rng = np.random.default_rng(seed)

    if venues is None:
        venues = _synthetic_venue_layer(rg_scale_km, rng)

    agents = sample_traversement_potential(
        n=n_agents,
        rg_scale_km=rg_scale_km,
        rg_growth_exponent=rg_growth_exponent,
        seed_hiv_prevalence=seed_hiv_prevalence,
        rng=rng,
    )

    trajectories = generate_walks(
        agents=agents,
        venues=venues,
        n_steps=n_steps,
        epr_rho=epr_rho,
        epr_gamma=epr_gamma,
        jump_length_exponent=jump_length_exponent,
        rng=rng,
    )

    colocs = colocations(trajectories, space_bin=space_bin, time_bin=time_bin)
    edges = sharing_edges(colocs, sharing_prob=sharing_prob, kappa_share=kappa_share, rng=rng)
    graph, stats = build_contact_graph(edges, n_agents=n_agents)

    return graph, stats, agents


def _synthetic_venue_layer(rg_scale_km: float, rng: np.random.Generator) -> List[Venue]:
    """
    Generate a minimal synthetic venue layer for testing.
    Real usage should load SSP locations from NASEN/AmFAR data.
    Drug market venues are PLACEHOLDER — labeled explicitly.
    """
    venues = []

    # SSP venues: 5 real-type attractors (high relevance)
    ssp_xs = rng.normal(0, rg_scale_km, 5)
    ssp_ys = rng.normal(0, rg_scale_km, 5)
    for i, (x, y) in enumerate(zip(ssp_xs, ssp_ys)):
        venues.append(Venue(venue_id=i, x=float(x), y=float(y),
                            relevance=1.5, venue_type="ssp"))

    # Drug market PLACEHOLDER venues: lower relevance, explicitly labeled
    # TODO: replace with real data source (overdose/EMS hotspots, arrest data,
    # ethnographic maps) — AC must approve before use in published results
    dm_xs = rng.normal(0, rg_scale_km * 0.7, 3)
    dm_ys = rng.normal(0, rg_scale_km * 0.7, 3)
    for i, (x, y) in enumerate(zip(dm_xs, dm_ys)):
        venues.append(Venue(venue_id=5 + i, x=float(x), y=float(y),
                            relevance=1.0, venue_type="drug_market_placeholder"))

    return venues
