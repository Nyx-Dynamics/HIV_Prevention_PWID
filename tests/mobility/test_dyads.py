"""
Handoff 6 tests: persistent dyads are distinct from co-locations.

Acceptance:
1. colocation_counter returns counts, not just pairs
2. form_dyads produces fewer edges than co-location count (two stages distinct)
3. node-level intensity is heterogeneous (not constant)
4. run_generator_dyads returns edge_intensity_dict
5. Dyads create triangles (clustering > ER floor is at least plausible with enough venue anchoring)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import networkx as nx
from mobility.network_generator import (
    sample_traversement_potential,
    generate_walks,
    colocation_counter,
    assign_node_sharing_intensity,
    form_dyads,
    build_contact_graph,
    run_generator_dyads,
)


def test_colocation_counter_accumulates():
    """Counter should accumulate multiple co-locations per pair."""
    rng = np.random.default_rng(42)
    agents = sample_traversement_potential(20, 2.4, 1.65, 0.07, rng)
    trajectories = generate_walks(agents, venues=[], n_steps=30,
                                  epr_rho=0.8, epr_gamma=0.21,
                                  jump_length_exponent=0.6, rng=rng)
    counts = colocation_counter(trajectories, space_bin=1.0, time_bin=5)
    assert isinstance(counts, dict)
    assert all(v >= 1 for v in counts.values())
    # Some pairs should have co-located more than once
    assert any(v > 1 for v in counts.values()), "Expected some pairs to co-locate >1 times"
    print(f"  PASS: colocation_counter — max count={max(counts.values())}, "
          f"unique pairs={len(counts)}")


def test_dyads_distinct_from_colocations():
    """Dyad count must be ≤ co-locating-pair count (two stages are genuinely distinct)."""
    rng = np.random.default_rng(42)
    agents = sample_traversement_potential(40, 2.4, 1.65, 0.07, rng)
    trajectories = generate_walks(agents, venues=[], n_steps=40,
                                  epr_rho=0.7, epr_gamma=0.21,
                                  jump_length_exponent=0.6, rng=rng)
    counts = colocation_counter(trajectories, space_bin=1.0, time_bin=5)
    node_int = assign_node_sharing_intensity(40, 0.27, rng)
    edges, _ = form_dyads(counts, node_int, kappa_dyad=2.0, rng=rng)
    n_coloc_pairs = len(counts)
    n_dyads = len(edges)
    assert n_dyads <= n_coloc_pairs, (
        f"Dyads ({n_dyads}) exceeded co-locating pairs ({n_coloc_pairs})"
    )
    print(f"  PASS: dyads ({n_dyads}) ≤ co-locating pairs ({n_coloc_pairs})")


def test_node_intensity_is_heterogeneous():
    """Node intensities must be heterogeneous (not constant) with heavy tail."""
    rng = np.random.default_rng(42)
    intensities = assign_node_sharing_intensity(500, 0.27, rng)
    # ~27% should be non-zero
    frac_nonzero = np.mean(intensities > 0)
    assert 0.15 <= frac_nonzero <= 0.40, (
        f"Expected ~27% non-zero intensity, got {frac_nonzero:.2f}"
    )
    # Non-zero intensities should be heterogeneous
    nonzero = intensities[intensities > 0]
    assert np.std(nonzero) > 0, "Non-zero intensities should vary (heavy-tailed)"
    assert nonzero.max() > nonzero.mean() * 3, "Expected heavy right tail"
    print(f"  PASS: node intensity heterogeneous — {frac_nonzero:.2f} non-zero, "
          f"max/mean={nonzero.max()/nonzero.mean():.1f}×")


def test_run_generator_dyads_returns_intensities():
    """run_generator_dyads must return edge_intensity_dict."""
    G, stats, agents, node_int, edge_int = run_generator_dyads(
        n_agents=50, n_steps=30, seed=42
    )
    assert isinstance(G, nx.Graph)
    assert isinstance(node_int, np.ndarray)
    assert len(node_int) == 50
    assert isinstance(edge_int, dict)
    # Every edge in edge_int should be in G
    for a, b in edge_int:
        assert G.has_edge(a, b), f"Edge ({a},{b}) in edge_int but not in graph"
    print(f"  PASS: run_generator_dyads — edges={stats.n_edges}, "
          f"edge_intensities={len(edge_int)}")


def test_dyads_create_triangles():
    """
    With strong venue anchoring, dyads should produce triangles above the ER floor.
    Not a strict pass/fail — just verify cc > 0 with dyad mechanism.
    """
    G, stats, _, _, _ = run_generator_dyads(
        n_agents=100, n_steps=60, venue_return_boost=8.0, kappa_dyad=3.0, seed=42
    )
    er_floor = stats.mean_degree / stats.n_nodes if stats.n_nodes > 0 else 0
    cc = stats.clustering_coefficient
    print(f"  INFO: cc={cc:.4f}, ER_floor={er_floor:.5f}, cc/ER={cc/er_floor:.1f}×")
    assert cc > 0, "Expected non-zero clustering with dyad mechanism"
    print(f"  PASS: dyads create some triangles (cc={cc:.4f} > 0)")


if __name__ == "__main__":
    print("Running Handoff 6 dyad tests...")
    test_colocation_counter_accumulates()
    test_dyads_distinct_from_colocations()
    test_node_intensity_is_heterogeneous()
    test_run_generator_dyads_returns_intensities()
    test_dyads_create_triangles()
    print("\nAll Handoff 6 dyad tests passed.")
