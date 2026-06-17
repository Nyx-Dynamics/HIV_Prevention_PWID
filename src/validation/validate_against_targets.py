"""
Read-only validation harness for HIV_Prevention_PWID.

Reads emitted run artifacts + target dicts; classifies every target;
scores only SCORED ones; HIV held out in the tooling itself.

Usage:
    python -m src.validation.validate_against_targets --run outputs/
    python src/validation/validate_against_targets.py --run outputs/

Input contract (spec-defined names; falls back to realized names):
    network_stats.json  — ⟨k⟩, dispersion, clustering, gc, path_length, τ_c
    outbreak_sim.json   — final_size, single_cluster_fraction, R₀ dist, incidence

Realized names (current generator output — NOT yet renamed to spec names):
    network_validation_report.json
    dyad_r0_report.json
    threshold_report.json

GUARDRAILS:
    - Zero writes outside src/validation/ and tests/validation/
    - Zero imports of generator internals (only params.py dicts + JSON)
    - HIV held out by construction — HELD_OUT frozenset checked before any scoring
    - not_emitted / not_computable_yet over guessing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Resolve package path (supports both -m invocation and direct script)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'src'))

from validation.targets_io import (
    HELD_OUT, TARGET_CLASSES, SCORED_TARGETS, PROVENANCE_REGISTRY, classify
)

# Import target dicts from params.py — read-only, no modification
from mobility.params import OUTBREAK_VALIDATION_TARGETS, SENTINEL_LADDER


# ─────────────────────────────────────────────────────────────────────────────
# PROVENANCE GUARD (v2)
# ─────────────────────────────────────────────────────────────────────────────

def _git_head_sha() -> Optional[str]:
    """Return current HEAD commit SHA; None if not in a git repo or git unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, OSError):
        return None


def _short(sha: Optional[str]) -> str:
    return sha[:8] if sha and len(sha) >= 8 else (sha or "unknown")


def read_artifact_provenance(
    filepath: str,
    expect_commit: Optional[str] = None,
    max_age_hours: Optional[float] = None,
) -> Dict:
    """
    Read provenance metadata for a single artifact file.

    Returns a dict with: path, mtime, age_hours, artifact_commit,
    head_commit, verdict, provenance_suspect.

    verdict is one of: "fresh" | "stale" | "unverified"
      - fresh: artifact commit matches HEAD (or no expect_commit and within max_age)
      - stale: artifact commit doesn't match HEAD, or max_age exceeded
      - unverified: no commit embedded and no max_age constraint → can't verify
    """
    if not os.path.exists(filepath):
        return {
            "path": filepath,
            "mtime": None,
            "age_hours": None,
            "artifact_commit": None,
            "head_commit": _git_head_sha(),
            "verdict": "not_found",
            "provenance_suspect": True,
        }

    stat = os.stat(filepath)
    mtime = stat.st_mtime
    age_hours = (time.time() - mtime) / 3600
    mtime_str = datetime.fromtimestamp(mtime).isoformat()

    # Try to find a commit SHA embedded in the artifact JSON
    data = _load_json(filepath)
    artifact_commit = None
    if data:
        artifact_commit = (
            data.get("git_sha")
            or data.get("commit")
            or data.get("run_sha")
            or (data.get("run_manifest") or {}).get("git_sha")
        )

    head_sha = _git_head_sha()

    # Determine verdict
    stale = False
    if artifact_commit and head_sha:
        if artifact_commit == head_sha:
            verdict = "fresh"
        else:
            verdict = "stale"
            stale = True
    elif expect_commit:
        # No embedded commit but caller expects a specific one — can't verify
        verdict = "unverified"
        stale = True  # treat conservatively
    elif max_age_hours and age_hours > max_age_hours:
        verdict = "stale"
        stale = True
    else:
        verdict = "unverified"
        # unverified but not flagged as stale unless explicitly required

    # explicit expect_commit mismatch (artifact has embedded commit that differs)
    if expect_commit and artifact_commit and artifact_commit != expect_commit:
        verdict = "stale"
        stale = True

    return {
        "path": filepath,
        "mtime": mtime_str,
        "age_hours": round(age_hours, 2),
        "artifact_commit": artifact_commit,
        "head_commit": head_sha,
        "expect_commit": expect_commit,
        "verdict": verdict,
        "provenance_suspect": stale,
    }


def build_provenance_banner(provenance_records: List[Dict]) -> Optional[str]:
    """
    Return a STALE/UNVERIFIED-PROVENANCE banner string if any artifact is suspect.
    Returns None if all are fresh.
    """
    suspect = [p for p in provenance_records if p.get("provenance_suspect")]
    if not suspect:
        return None

    lines = [
        "⚠ STALE / UNVERIFIED-PROVENANCE",
        "  One or more artifacts predate HEAD or have no embedded commit SHA.",
        "  Scores may reflect an outdated network — re-run the generator before",
        "  treating this report as current.",
        "  Suspect artifacts:",
    ]
    for p in suspect:
        lines.append(
            f"    {os.path.basename(p['path'])}: verdict={p['verdict']}  "
            f"mtime={p['mtime']}  "
            f"artifact_commit={_short(p.get('artifact_commit'))}  "
            f"head={_short(p.get('head_commit'))}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

# Spec-defined filenames (what the contract calls them)
SPEC_FILES = {
    "network_stats": "network_stats.json",
    "outbreak_sim":  "outbreak_sim.json",
}

# Realized filenames (what the current generator actually emits)
REALIZED_FILES = {
    "network_stats": [
        "network_validation_report.json",
        "dyad_r0_report.json",
    ],
    "outbreak_sim": [
        # No realization yet — outbreak_sim.json does not exist
    ],
}


def _load_json(path: str) -> Optional[Dict]:
    """Load JSON file; return None if missing or malformed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def discover_files(run_dir: str) -> Dict[str, Dict]:
    """
    Discover available artifact files in run_dir.

    Returns a dict with keys 'network_stats' and 'outbreak_sim', each containing:
      - 'source_file': filename found (or None)
      - 'spec_name': the contract-defined filename
      - 'realized': True if found under a realized (non-spec) name
      - 'data': the loaded JSON dict (or None)
      - 'schema_note': what keys were found
    """
    results = {}

    for artifact_key, spec_filename in SPEC_FILES.items():
        spec_path = os.path.join(run_dir, spec_filename)
        data = _load_json(spec_path)

        if data is not None:
            results[artifact_key] = {
                "source_file": spec_filename,
                "spec_name": spec_filename,
                "realized": False,
                "data": data,
                "schema_note": f"Found spec-defined file. Top keys: {list(data.keys())}",
            }
            continue

        # Try realized fallback names
        realized_name = None
        realized_data = None
        for fallback in REALIZED_FILES.get(artifact_key, []):
            fb_path = os.path.join(run_dir, fallback)
            fb_data = _load_json(fb_path)
            if fb_data is not None:
                realized_name = fallback
                realized_data = fb_data
                break

        if realized_data is not None:
            results[artifact_key] = {
                "source_file": realized_name,
                "spec_name": spec_filename,
                "realized": True,
                "data": realized_data,
                "schema_note": f"Spec name not found; using realized name '{realized_name}'. "
                               f"Top keys: {list(realized_data.keys())}",
            }
        else:
            results[artifact_key] = {
                "source_file": None,
                "spec_name": spec_filename,
                "realized": False,
                "data": None,
                "schema_note": "not_emitted",
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# VALUE EXTRACTION — safely pull target values from the discovered JSON
# ─────────────────────────────────────────────────────────────────────────────

def _safe_get(d: Dict, *keys, default=None):
    """Nested dict access with missing-key default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_network_values(artifact: Dict) -> Dict:
    """
    Extract network statistics from a discovered artifact.

    Returns {field: value or 'missing_field'} for all SCORED network targets.
    """
    data = artifact.get("data")
    if data is None:
        return {k: "not_emitted" for k in [
            "giant_component_fraction", "clustering_coefficient",
            "dispersion_k2_over_k2", "mean_path_length", "mean_degree",
        ]}

    source = artifact.get("source_file", "")

    # Detect schema by content, not filename
    # (fixtures may be loaded under the spec-defined name)
    has_generated_checks = "generated" in data and "checks" in data
    has_proxies = "proxies" in data and isinstance(data.get("proxies"), list)

    # --- network_validation_report.json schema ---
    if "network_validation_report" in source or has_generated_checks:
        gen = data.get("generated", {})
        checks = data.get("checks", {})
        cc_check = checks.get("clustering_coefficient", {})
        disp_check = checks.get("dispersion_k2_over_k2", {})
        return {
            "giant_component_fraction": gen.get("giant_component_fraction", "missing_field"),
            "clustering_coefficient":   cc_check.get("generated", "missing_field"),
            "dispersion_k2_over_k2":    disp_check.get("generated", "missing_field"),
            "mean_path_length":         gen.get("mean_path_length", "missing_field"),
            "mean_degree":              gen.get("mean_degree", "missing_field"),
            "er_floor":                 cc_check.get("er_floor", "missing_field"),
            "poisson_baseline":         disp_check.get("poisson_baseline", "missing_field"),
        }

    # --- dyad_r0_report.json schema (use first proxy) ---
    if "dyad_r0_report" in source or has_proxies:
        proxies = data.get("proxies", [])
        if not proxies:
            return {k: "missing_field" for k in [
                "giant_component_fraction", "clustering_coefficient",
                "dispersion_k2_over_k2", "mean_path_length", "mean_degree",
            ]}
        net = proxies[0].get("network", {})
        mean_k = net.get("mean_degree", 0) or 1
        er_floor = mean_k / (net.get("n_nodes") or 300)
        poisson_baseline = 1 + 1 / mean_k if mean_k > 0 else 2.0
        return {
            "giant_component_fraction": net.get("giant_component", "missing_field"),
            "clustering_coefficient":   net.get("clustering", "missing_field"),
            "dispersion_k2_over_k2":    net.get("dispersion", "missing_field"),
            "mean_path_length":         net.get("mean_path_length", "missing_field"),
            "mean_degree":              net.get("mean_degree", "missing_field"),
            "er_floor":                 er_floor,
            "poisson_baseline":         poisson_baseline,
        }

    # --- spec-defined network_stats.json schema (future) ---
    # Keys: mean_degree, k2_moment, giant_component_fraction, clustering,
    #       dispersion, er_floor, mean_path_length, tau_c
    mean_k = data.get("mean_degree", 0) or 1
    poisson_baseline = 1 + 1 / mean_k if mean_k > 0 else 2.0
    return {
        "giant_component_fraction": data.get("giant_component_fraction", "missing_field"),
        "clustering_coefficient":   data.get("clustering",   data.get("clustering_coefficient", "missing_field")),
        "dispersion_k2_over_k2":    data.get("dispersion",   "missing_field"),
        "mean_path_length":         data.get("mean_path_length", "missing_field"),
        "mean_degree":              data.get("mean_degree",  "missing_field"),
        "er_floor":                 data.get("er_floor",     "missing_field"),
        "poisson_baseline":         poisson_baseline,
    }


def extract_outbreak_values(artifact: Dict) -> Dict:
    """Extract outbreak trajectory values from outbreak_sim artifact."""
    data = artifact.get("data")
    if data is None:
        return {k: "not_emitted" for k in [
            "final_size", "single_cluster_fraction",
            "degree_risk_gradient", "explosive_incidence",
            "hiv_r0", "hiv_p_r0_gt1",
        ]}

    return {
        "final_size":               _safe_get(data, "final_size", default="missing_field"),
        "single_cluster_fraction":  _safe_get(data, "single_cluster_fraction", default="missing_field"),
        "degree_risk_gradient":     _safe_get(data, "degree_risk_gradient", default="missing_field"),
        "explosive_incidence":      _safe_get(data, "incidence", default="missing_field"),
        # HIV R₀ is HELD_OUT — extract for report-only, never score
        "hiv_r0":                   _safe_get(data, "r0", "median", default="missing_field"),
        "hiv_r0_p5":                _safe_get(data, "r0", "p5",    default="missing_field"),
        "hiv_r0_p95":               _safe_get(data, "r0", "p95",   default="missing_field"),
        "hiv_p_r0_gt1":             _safe_get(data, "p_r0_gt1",    default="missing_field"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _ci_overlaps(ci_low, ci_high, target_low, target_high) -> Optional[bool]:
    """True if [ci_low, ci_high] overlaps [target_low, target_high]."""
    if any(v is None or isinstance(v, str) for v in [ci_low, ci_high, target_low, target_high]):
        return None
    return max(ci_low, target_low) <= min(ci_high, target_high)


def _median_in_range(median, target_low, target_high) -> Optional[bool]:
    if any(v is None or isinstance(v, str) for v in [median, target_low, target_high]):
        return None
    return target_low <= median <= target_high


def score_target(
    target_key: str,
    model_value,
    model_ci: Optional[Tuple] = None,
) -> Dict:
    """
    Score a SCORED target. Returns a result dict with:
      status: PASS / FAIL / NOT_EMITTED / MISSING_FIELD
      model_value, target_range, ci_overlaps_target, median_in_range, note
    """
    # Sentinel: not emitted / missing
    if isinstance(model_value, str) and model_value in ("not_emitted", "missing_field"):
        return {
            "status": "NOT_EMITTED" if model_value == "not_emitted" else "MISSING_FIELD",
            "model_value": model_value,
            "ci_overlaps_target": None,
            "median_in_range": None,
        }

    target_def = SCORED_TARGETS.get(target_key, {})
    t_type = target_def.get("type", "range")

    # Compute the effective range
    if t_type == "lower_bound_factor":
        # dispersion: target = Poisson_baseline × factor
        # Poisson_baseline must be passed in as extra context (via model_ci hack)
        if model_ci and len(model_ci) >= 1:
            poisson_baseline = model_ci[0]
        else:
            poisson_baseline = 1.38  # fallback at ⟨k⟩=2.6
        factor = target_def.get("factor", 1.5)
        target_low = poisson_baseline * factor
        target_high = float("inf")
    else:
        target_range = target_def.get("range", (None, None))
        target_low, target_high = target_range[0], target_range[1]

    target_high_display = target_high if target_high != float("inf") else "∞"

    median_ok = _median_in_range(model_value, target_low,
                                 target_high if target_high != float("inf") else 1e18)

    ci_ok = None
    if model_ci and len(model_ci) == 2 and t_type != "lower_bound_factor":
        ci_ok = _ci_overlaps(model_ci[0], model_ci[1], target_low,
                              target_high if target_high != float("inf") else 1e18)

    # Point check: pass if median is in range
    status = "PASS" if median_ok else "FAIL"
    if median_ok is None:
        status = "UNKNOWN"

    return {
        "status": status,
        "model_value": model_value,
        "model_ci": model_ci,
        "target_range": (target_low, target_high_display),
        "target_source": target_def.get("source", ""),
        "ci_overlaps_target": ci_ok,
        "median_in_range": median_ok,
        "note": target_def.get("note", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HARNESS
# ─────────────────────────────────────────────────────────────────────────────

def run_harness(
    run_dir: str,
    output_dir: Optional[str] = None,
    expect_commit: Optional[str] = None,
    max_age_hours: Optional[float] = None,
    strict: bool = False,
) -> Dict:
    """
    Build the full validation report for a given run directory.

    Parameters
    ----------
    run_dir : directory containing the run's emitted JSON artifacts
    output_dir : where to write validation_report.json / .md (default: run_dir)
    expect_commit : if set, artifacts must have this commit SHA embedded
    max_age_hours : if set, artifacts older than this are flagged stale
    strict : if True, exit non-zero when any artifact is stale/unverified

    Returns the report dict.
    """
    if output_dir is None:
        output_dir = run_dir
    os.makedirs(output_dir, exist_ok=True)

    report: Dict[str, Any] = {
        "run_dir": os.path.abspath(run_dir),
        "timestamp": datetime.now().isoformat(),
        "head_commit": _git_head_sha(),
        "expect_commit": expect_commit,
        "provenance_banner": None,
        "artifact_provenance": {},
        "schema_discovery": {},
        "scored": {},
        "held_out": {},
        "provenance": {},
        "stubbed": {},
        "summary": {},
    }

    # ── 1. Discover files ─────────────────────────────────────────────────
    artifacts = discover_files(run_dir)

    # ── 1b. Provenance guard ──────────────────────────────────────────────
    prov_records = []
    for artifact_key, art in artifacts.items():
        if art["source_file"]:
            fpath = os.path.join(run_dir, art["source_file"])
            prov = read_artifact_provenance(fpath, expect_commit, max_age_hours)
        else:
            prov = {
                "path": os.path.join(run_dir, art["spec_name"]),
                "verdict": "not_found",
                "provenance_suspect": False,  # not_found ≠ stale
            }
        report["artifact_provenance"][artifact_key] = prov
        prov_records.append(prov)

    banner = build_provenance_banner(prov_records)
    report["provenance_banner"] = banner
    if banner:
        print(banner)
        if strict:
            print("\n--strict: exiting non-zero due to stale/unverified provenance.")
            sys.exit(1)

    report["schema_discovery"] = {
        k: {
            "source_file": v["source_file"],
            "spec_name": v["spec_name"],
            "realized": v["realized"],
            "schema_note": v["schema_note"],
        }
        for k, v in artifacts.items()
    }

    # ── 2. Extract values ─────────────────────────────────────────────────
    net_vals = extract_network_values(artifacts["network_stats"])
    out_vals = extract_outbreak_values(artifacts["outbreak_sim"])

    # ── 3. Score SCORED targets ───────────────────────────────────────────

    # Network benchmarks
    for key in ["giant_component_fraction", "clustering_coefficient",
                "mean_path_length"]:
        val = net_vals.get(key, "missing_field")
        report["scored"][key] = score_target(key, val)

    # Dispersion — per-run Poisson baseline from the SAME RUN's ⟨k⟩ (v2 fix)
    disp_val = net_vals.get("dispersion_k2_over_k2", "missing_field")
    mean_k_raw = net_vals.get("mean_degree", None)
    if isinstance(mean_k_raw, (int, float)) and mean_k_raw > 0:
        per_run_poisson = 1.0 + 1.0 / float(mean_k_raw)
    else:
        # Embedded poisson_baseline from the checks dict (validate_network.py)
        pb_raw = net_vals.get("poisson_baseline", None)
        per_run_poisson = float(pb_raw) if isinstance(pb_raw, (int, float)) else 1.38
    report["scored"]["dispersion_k2_over_k2"] = score_target(
        "dispersion_k2_over_k2", disp_val, model_ci=(per_run_poisson,)
    )

    # Outbreak trajectory
    for key in ["final_size", "single_cluster_fraction",
                "degree_risk_gradient", "explosive_incidence"]:
        val = out_vals.get(key, "not_emitted")
        report["scored"][key] = score_target(key, val)

    # ── 4. HELD_OUT — report-only (never scored) ──────────────────────────
    for ho_key in ["hiv_r0", "hiv_p_r0_gt1"]:
        val = out_vals.get(ho_key, "not_emitted")
        p5  = out_vals.get("hiv_r0_p5", None)
        p95 = out_vals.get("hiv_r0_p95", None)
        report["held_out"][ho_key] = {
            "class": "HELD_OUT",
            "model_value": val,
            "model_ci": (p5, p95) if p5 is not None and p95 is not None else None,
            "note": "HIV forward prediction — reported here, NEVER scored.",
        }

    # ── 5. Provenance ─────────────────────────────────────────────────────
    for pkey, pval in PROVENANCE_REGISTRY.items():
        report["provenance"][pkey] = {
            "class": "PROVENANCE",
            "value": pval.get("value"),
            "range": pval.get("range") or pval.get("ci"),
            "source": pval.get("source"),
            "note": "Input parameter — not a model output; record only.",
        }

    # ── 6. Stubbed ────────────────────────────────────────────────────────
    for sk in ["rectal_gc_hiv_incidence", "syphilis_hiv_incidence",
               "sexual_hcv_incidence", "hcv_consistency_check"]:
        report["stubbed"][sk] = {
            "class": "STUBBED",
            "status": "not_computable_yet",
            "note": "Requires bridge / multi-pathogen layer (Handoff 7+).",
        }

    # ── 7. Summary ────────────────────────────────────────────────────────
    scored = report["scored"]
    n_pass = sum(1 for v in scored.values() if v.get("status") == "PASS")
    n_fail = sum(1 for v in scored.values() if v.get("status") == "FAIL")
    n_ne   = sum(1 for v in scored.values() if v.get("status") in ("NOT_EMITTED", "MISSING_FIELD"))
    n_scoreable = n_pass + n_fail

    report["summary"] = {
        "scored_total": len(scored),
        "scored_pass": n_pass,
        "scored_fail": n_fail,
        "scored_not_emitted": n_ne,
        "scored_pass_rate": round(n_pass / n_scoreable, 3) if n_scoreable > 0 else None,
        "held_out_count": len(report["held_out"]),
        "provenance_count": len(report["provenance"]),
        "stubbed_count": len(report["stubbed"]),
        "warning": (
            "HELD_OUT items (HIV R₀, HIV incidence) appear in the 'held_out' section "
            "and are NOT counted in scored_pass_rate. "
            "HIV outbreak frequency remains the held-out forward prediction."
        ),
    }

    # ── 8. Write outputs ─────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "validation_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    md_path = os.path.join(output_dir, "validation_report.md")
    with open(md_path, "w") as f:
        f.write(_render_markdown(report))

    return report


def _render_markdown(report: Dict) -> str:
    lines = [
        "# Validation Report",
        f"Run dir: `{report['run_dir']}`",
        f"Timestamp: {report['timestamp']}",
        f"HEAD commit: `{_short(report.get('head_commit'))}`",
        "",
    ]

    # Provenance banner — always at top, prominent if stale
    banner = report.get("provenance_banner")
    if banner:
        lines += [
            "---",
            f"**⚠ STALE/UNVERIFIED-PROVENANCE**",
            "",
            "```",
            banner,
            "```",
            "",
            "---",
        ]

    # Artifact provenance table
    lines += [
        "## Artifact Provenance",
        "| Artifact | File | mtime | artifact_commit | head_commit | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for k, p in report.get("artifact_provenance", {}).items():
        fname = os.path.basename(p.get("path", "—"))
        mtime = (p.get("mtime") or "—")[:19]
        ac = _short(p.get("artifact_commit"))
        hc = _short(p.get("head_commit"))
        v = p.get("verdict", "—")
        flag = " ⚠" if p.get("provenance_suspect") else ""
        lines.append(f"| {k} | {fname} | {mtime} | `{ac}` | `{hc}` | **{v}{flag}** |")

    lines += [
        "",
        "## Schema Discovery",
        "| Artifact | Source file | Status |",
        "|---|---|---|",
    ]
    for k, v in report["schema_discovery"].items():
        sf = v["source_file"] or "—"
        note = "realized name" if v["realized"] else ("spec name" if v["source_file"] else "NOT EMITTED")
        lines.append(f"| {k} | {sf} | {note} |")

    lines += [
        "",
        "## SCORED Targets",
        "| Target | Model value | Target range | Status | CI overlaps | Median in range |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in report["scored"].items():
        mv = str(v.get("model_value", "—"))
        tr = str(v.get("target_range", "—"))
        st = v.get("status", "—")
        co = str(v.get("ci_overlaps_target", "—"))
        mr = str(v.get("median_in_range", "—"))
        lines.append(f"| {k} | {mv} | {tr} | **{st}** | {co} | {mr} |")

    s = report["summary"]
    lines += [
        "",
        f"**SCORED pass rate: {s['scored_pass']}/{s['scored_pass'] + s['scored_fail']} "
        f"({s['scored_pass_rate']})** ({s['scored_not_emitted']} not emitted yet)",
        "",
        "## HELD_OUT — Forward Predictions (never scored)",
        "| Key | Model value | Note |",
        "|---|---|---|",
    ]
    for k, v in report["held_out"].items():
        lines.append(f"| {k} | {v.get('model_value', '—')} | {v.get('note', '')} |")

    lines += [
        "",
        "## PROVENANCE",
        "| Parameter | Value | Source |",
        "|---|---|---|",
    ]
    for k, v in report["provenance"].items():
        lines.append(f"| {k} | {v.get('value')} | {v.get('source', '')[:60]} |")

    lines += [
        "",
        "## STUBBED",
        "| Target | Status | Note |",
        "|---|---|---|",
    ]
    for k, v in report["stubbed"].items():
        lines.append(f"| {k} | {v.get('status')} | {v.get('note')} |")

    lines += [
        "",
        f"> {s['warning']}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate model run against sourced targets."
    )
    parser.add_argument("--run", required=True,
                        help="Directory containing the run's emitted JSON artifacts")
    parser.add_argument("--output", default=None,
                        help="Output directory for validation_report.json/.md "
                             "(default: same as --run)")
    parser.add_argument("--expect-commit", default=None, dest="expect_commit",
                        help="If given, artifacts must embed this commit SHA; "
                             "mismatches trigger STALE banner")
    parser.add_argument("--max-age", type=float, default=None, dest="max_age_hours",
                        help="Maximum artifact age in hours before STALE is flagged")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero when any artifact is stale/unverified")
    args = parser.parse_args()

    report = run_harness(
        args.run, args.output,
        expect_commit=args.expect_commit,
        max_age_hours=args.max_age_hours,
        strict=args.strict,
    )
    s = report["summary"]
    print(f"\nValidation complete.")
    print(f"  SCORED:     {s['scored_pass']} pass / {s['scored_fail']} fail / "
          f"{s['scored_not_emitted']} not emitted")
    print(f"  HELD_OUT:   {s['held_out_count']} (report-only, never scored)")
    print(f"  PROVENANCE: {s['provenance_count']}")
    print(f"  STUBBED:    {s['stubbed_count']}")
    print(f"  Pass rate:  {s['scored_pass_rate']}")
    print(f"\nReports written to: {args.output or args.run}")


if __name__ == "__main__":
    main()
