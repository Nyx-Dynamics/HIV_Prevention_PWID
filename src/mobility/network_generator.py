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


def _venue_proximity_weight(x: float, y: float, venues: List[Venue], capture_radius: float) -> float:
    """Return venue relevance if (x,y) is within capture_radius of a venue; else 1.0."""
    if not venues:
        return 1.0
    for v in venues:
        if (x - v.x) ** 2 + (y - v.y) ** 2 <= capture_radius ** 2:
            return float(v.relevance)
    return 1.0


def generate_walks(
    agents: List[Agent],
    venues: List[Venue],
    n_steps: int,
    epr_rho: float,
    epr_gamma: float,
    jump_length_exponent: float,
    rng: np.random.Generator,
    venue_return_boost: float = 1.0,
) -> List[Trajectory]:
    """
    EPR walk for each agent over n_steps.

    venue_return_boost: multiply visit-count weight by this factor for
    locations near a venue during preferential-return step. Values > 1
    increase clustering at venues → heavier-tailed degree distribution.
    Default 1.0 = original behavior (no venue boost in return).

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
                # Preferential return: return to a previously visited location,
                # with venue locations up-weighted by venue_return_boost.
                locs = list(location_counts.keys())
                capture_r = agent.rg * 0.15
                counts = np.array([
                    location_counts[l] * (
                        _venue_proximity_weight(l[0], l[1], venues, capture_r)
                        * venue_return_boost if venue_return_boost > 1.0 else location_counts[l]
                    )
                    for l in locs
                ], dtype=float)
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
# PERSISTENT DYADS (Task 1–3 of Handoff 6)
# Distinct from sharing_edges: accumulated co-location count drives dyad
# formation; node-level intensity is the structural fix for clustering.
# ─────────────────────────────────────────────────────────────────────────────

def colocation_counter(
    trajectories: List[Trajectory],
    space_bin: float = 0.5,
    time_bin: int = 5,
) -> Dict[Tuple[int, int], int]:
    """
    Count co-location events per unique agent pair.

    Returns {(a, b): n_colocations} with a < b (canonical order).
    Distinct from colocations() which returns a flat list; this accumulates.
    """
    from collections import Counter
    counter: Dict[Tuple[int, int], int] = {}

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

    for agents_in_bin in bin_map.values():
        unique = list(set(agents_in_bin))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = (min(unique[i], unique[j]), max(unique[i], unique[j]))
                counter[pair] = counter.get(pair, 0) + 1

    return counter


def assign_node_sharing_intensity(
    n_agents: int,
    sharing_prevalence: float,
    rng: np.random.Generator,
    shape: float = 0.5,
    scale: float = 0.05,
) -> np.ndarray:
    """
    Task 2: Assign each node a sharing intensity from a mixture distribution.

    Two-component mixture (not a global constant — node-level heterogeneity
    creates correlated edges and thereby closes triangles):
      - With probability (1 − sharing_prevalence): zero intensity (non-sharer)
      - With probability sharing_prevalence: Gamma(shape, scale) intensity

    CALIBRATION:
      - sharing_prevalence = 0.27 from NHBS any-sharing 12-mo prevalence
        (Burnett JC et al. MMWR 67(1) 2018. DOI 10.15585/mmwr.mm6701a5).
      - Within-sharer intensity (frequency given any sharing): HANDOFF 8B RESULT:
        The NHBS MMWR 67(1) publication does NOT contain the frequency breakdown
        (every time / >half / <half). That data is in the NHBS 2015 public-use
        dataset (CDC NHBS data portal), which requires direct access — not codeable
        from published tables. Per the 8B guardrail: FLAGGED AND STOPPED.
        Gamma(shape=0.5, scale=0.05) remains a PLACEHOLDER until AC retrieves
        the public-use dataset or approves an alternative source.
        PENDING AC: access CDC NHBS public-use data or approve placeholder.

    Node-level (not per-edge) is the structural fix: high-intensity nodes
    form edges with many venue co-visitors → triangles → clustering.
    """
    is_sharer = rng.random(n_agents) < sharing_prevalence
    intensities = rng.gamma(shape=shape, scale=scale, size=n_agents)
    return intensities * is_sharer.astype(float)


def form_dyads(
    coloc_counts: Dict[Tuple[int, int], int],
    node_intensities: np.ndarray,
    kappa_dyad: float = 2.0,
    rng: np.random.Generator = None,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Task 1+2: Form persistent sharing dyads from accumulated co-locations
    and node-level sharing intensities.

    A dyad forms when both partners have non-zero intensity AND the
    accumulated co-location × geometric-mean-intensity exceeds a threshold.

    P(dyad | n_coloc, s_i, s_j) = 1 − exp(−kappa_dyad × n_coloc × √(s_i × s_j))

    Distinct from sharing_edges():
    - sharing_edges: independent Bernoulli per co-location event, global p
    - form_dyads: accumulated counts, node-level intensity, no global p

    Returns (edge_list, edge_intensity_dict) where edge_intensity is the
    geometric mean of the two nodes' propensities — used for per-edge T.

    kappa_dyad : calibration scalar; default 2.0 tuned to produce ⟨k⟩≈2.6
    with typical co-location counts and NHBS intensity parameters.
    ANCHOR WARNING: kappa_dyad is effectively the calibration knob for ⟨k⟩.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    edges: List[Tuple[int, int]] = []
    edge_intensities: Dict[Tuple[int, int], float] = {}

    for (a, b), n_coloc in coloc_counts.items():
        si = node_intensities[a]
        sj = node_intensities[b]
        if si <= 0 or sj <= 0:
            continue  # at least one non-sharer — no dyad
        dyad_intensity = float(np.sqrt(si * sj))  # geometric mean
        p_form = 1.0 - np.exp(-kappa_dyad * n_coloc * dyad_intensity)
        if rng.random() < p_form:
            edge = (min(a, b), max(a, b))
            edges.append(edge)
            edge_intensities[edge] = dyad_intensity

    return edges, edge_intensities


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
    venue_return_boost: float = 1.0,
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
        venue_return_boost=venue_return_boost,
    )

    colocs = colocations(trajectories, space_bin=space_bin, time_bin=time_bin)
    edges = sharing_edges(colocs, sharing_prob=sharing_prob, kappa_share=kappa_share, rng=rng)
    graph, stats = build_contact_graph(edges, n_agents=n_agents)

    return graph, stats, agents


def run_generator_dyads(
    n_agents: int = 300,
    n_steps: int = 80,
    rg_scale_km: float = 2.4,
    rg_growth_exponent: float = 1.65,
    seed_hiv_prevalence: float = 0.07,
    epr_rho: float = 0.60,
    epr_gamma: float = 0.21,
    jump_length_exponent: float = 0.60,
    sharing_prevalence: float = 0.27,
    intensity_shape: float = 0.5,
    intensity_scale: float = 0.05,
    kappa_dyad: float = 2.0,
    space_bin: float = 0.5,
    time_bin: int = 5,
    venues: Optional[List[Venue]] = None,
    venue_return_boost: float = 5.0,
    seed: int = 42,
) -> Tuple[nx.Graph, NetworkStats, List[Agent], np.ndarray, Dict]:
    """
    Dyad-based pipeline (Handoff 6). Replaces the per-co-location Bernoulli
    with persistent dyad formation driven by accumulated co-location counts and
    node-level heterogeneous sharing intensity.

    Returns (graph, stats, agents, node_intensities, edge_intensity_dict).
    node_intensities[i] = agent i's sharing propensity (0 = non-sharer).
    edge_intensity_dict[(a,b)] = geometric-mean intensity for that dyad.

    kappa_dyad is the calibration knob for ⟨k⟩ (analogous to kappa_share).
    Calibrated default 2.0 gives ⟨k⟩ ≈ 2–3 for typical inputs.
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
        venue_return_boost=venue_return_boost,
    )

    coloc_counts = colocation_counter(trajectories, space_bin=space_bin, time_bin=time_bin)

    node_intensities = assign_node_sharing_intensity(
        n_agents=n_agents,
        sharing_prevalence=sharing_prevalence,
        rng=rng,
        shape=intensity_shape,
        scale=intensity_scale,
    )

    edges, edge_intensities = form_dyads(coloc_counts, node_intensities, kappa_dyad, rng)
    graph, stats = build_contact_graph(edges, n_agents=n_agents)

    return graph, stats, agents, node_intensities, edge_intensities


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
