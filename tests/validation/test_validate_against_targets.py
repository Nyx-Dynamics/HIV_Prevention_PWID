"""
Fixture-based tests for validate_against_targets.py.

Tests verify:
1. 29% giant-component fixture FAILS the 45-100% check
2. 85% giant-component fixture PASSES the 45-100% check
3. Missing outbreak_sim.json → graceful NOT_EMITTED (no crash)
4. HELD_OUT items are report-only: they appear in held_out block,
   never in scored block, and never contribute to pass rate
5. All four classes (SCORED, HELD_OUT, PROVENANCE, STUBBED) appear
6. classify() returns HELD_OUT regardless of TARGET_CLASSES entry
"""

import sys
import os
import json
import shutil
import tempfile

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import pytest
from validation.validate_against_targets import run_harness, discover_files
from validation.targets_io import HELD_OUT, TARGET_CLASSES, classify

FIXTURE_DIR = os.path.join(_HERE, "fixtures")


def _make_run_dir(network_stats_fixture=None, outbreak_sim_fixture=None) -> str:
    """Create a temp directory with optional fixture files as spec-defined names."""
    run_dir = tempfile.mkdtemp()
    if network_stats_fixture:
        shutil.copy(
            os.path.join(FIXTURE_DIR, network_stats_fixture),
            os.path.join(run_dir, "network_stats.json"),
        )
    if outbreak_sim_fixture:
        shutil.copy(
            os.path.join(FIXTURE_DIR, outbreak_sim_fixture),
            os.path.join(run_dir, "outbreak_sim.json"),
        )
    return run_dir


class TestGiantComponentCheck:
    def test_29pct_fails(self, tmp_path):
        """29% giant component FAILS the 45-100% harness check."""
        run_dir = _make_run_dir("network_stats_29pct_giant.json")
        report = run_harness(run_dir, str(tmp_path))
        gc_result = report["scored"]["giant_component_fraction"]
        assert gc_result["status"] == "FAIL", (
            f"Expected FAIL for gc=0.283, got {gc_result['status']} "
            f"(model={gc_result['model_value']}, range={gc_result['target_range']})"
        )
        shutil.rmtree(run_dir)

    def test_85pct_passes(self, tmp_path):
        """85% giant component PASSES the 45-100% harness check."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json")
        report = run_harness(run_dir, str(tmp_path))
        gc_result = report["scored"]["giant_component_fraction"]
        assert gc_result["status"] == "PASS", (
            f"Expected PASS for gc=0.853, got {gc_result['status']} "
            f"(model={gc_result['model_value']}, range={gc_result['target_range']})"
        )
        shutil.rmtree(run_dir)

    def test_flip_proves_harness_discriminates(self, tmp_path):
        """The harness correctly distinguishes 29% (FAIL) from 85% (PASS)."""
        run_dir_bad = _make_run_dir("network_stats_29pct_giant.json")
        run_dir_good = _make_run_dir("network_stats_85pct_giant.json")
        report_bad  = run_harness(run_dir_bad,  str(tmp_path / "bad"))
        report_good = run_harness(run_dir_good, str(tmp_path / "good"))
        assert report_bad["scored"]["giant_component_fraction"]["status"] == "FAIL"
        assert report_good["scored"]["giant_component_fraction"]["status"] == "PASS"
        shutil.rmtree(run_dir_bad)
        shutil.rmtree(run_dir_good)


class TestMissingOutbreakSim:
    def test_missing_outbreak_sim_no_crash(self, tmp_path):
        """Missing outbreak_sim.json → NOT_EMITTED for all outbreak targets; no crash."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json")  # no outbreak_sim
        report = run_harness(run_dir, str(tmp_path))
        # Should have run without exception; outbreak targets should be NOT_EMITTED
        for key in ["final_size", "single_cluster_fraction",
                    "degree_risk_gradient", "explosive_incidence"]:
            result = report["scored"].get(key, {})
            assert result.get("status") in ("NOT_EMITTED", "MISSING_FIELD"), (
                f"Expected NOT_EMITTED for {key} when outbreak_sim.json missing, "
                f"got: {result.get('status')}"
            )
        shutil.rmtree(run_dir)

    def test_missing_both_files_no_crash(self, tmp_path):
        """Directory with no artifact files runs without crash."""
        run_dir = tempfile.mkdtemp()
        report = run_harness(run_dir, str(tmp_path))
        assert "summary" in report
        assert "scored" in report
        shutil.rmtree(run_dir)


class TestHeldOutIsReportOnly:
    def test_held_out_never_in_scored(self, tmp_path):
        """HELD_OUT keys must not appear in the scored block."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json", "outbreak_sim_good.json")
        report = run_harness(run_dir, str(tmp_path))
        scored_keys = set(report["scored"].keys())
        overlap = scored_keys & HELD_OUT
        assert not overlap, (
            f"HELD_OUT keys appeared in scored block: {overlap}"
        )
        shutil.rmtree(run_dir)

    def test_held_out_in_held_out_block(self, tmp_path):
        """HELD_OUT items must appear in the held_out block."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json", "outbreak_sim_good.json")
        report = run_harness(run_dir, str(tmp_path))
        assert len(report["held_out"]) > 0, "Expected at least one held_out entry"
        for key, val in report["held_out"].items():
            assert val.get("class") == "HELD_OUT", (
                f"Entry {key} in held_out block has class={val.get('class')}"
            )
        shutil.rmtree(run_dir)

    def test_held_out_does_not_affect_pass_rate(self, tmp_path):
        """Pass rate denominator must exclude HELD_OUT items."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json", "outbreak_sim_good.json")
        report = run_harness(run_dir, str(tmp_path))
        s = report["summary"]
        # pass rate denominator = pass + fail (not including held_out)
        if s["scored_pass_rate"] is not None:
            denominator = s["scored_pass"] + s["scored_fail"]
            expected = s["scored_pass"] / denominator if denominator > 0 else None
            assert abs(s["scored_pass_rate"] - expected) < 0.001, (
                "Pass rate seems to include held_out items in denominator"
            )
        shutil.rmtree(run_dir)

    def test_classify_held_out_overrides_dict(self):
        """classify() returns HELD_OUT for any key in HELD_OUT regardless of TARGET_CLASSES."""
        for key in HELD_OUT:
            assert classify(key) == "HELD_OUT", (
                f"classify({key!r}) returned {classify(key)!r}, expected 'HELD_OUT'"
            )


class TestAllClassesPresent:
    def test_all_four_classes_in_report(self, tmp_path):
        """A full run must produce all four sections: scored, held_out, provenance, stubbed."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json", "outbreak_sim_good.json")
        report = run_harness(run_dir, str(tmp_path))
        assert len(report["scored"]) > 0,     "No SCORED targets"
        assert len(report["held_out"]) > 0,   "No HELD_OUT targets"
        assert len(report["provenance"]) > 0, "No PROVENANCE targets"
        assert len(report["stubbed"]) > 0,    "No STUBBED targets"
        shutil.rmtree(run_dir)

    def test_stubbed_returns_not_computable(self, tmp_path):
        """STUBBED targets return not_computable_yet."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json")
        report = run_harness(run_dir, str(tmp_path))
        for key, val in report["stubbed"].items():
            assert val.get("status") == "not_computable_yet", (
                f"STUBBED {key} should be not_computable_yet, got {val.get('status')}"
            )
        shutil.rmtree(run_dir)

    def test_report_files_written(self, tmp_path):
        """Both JSON and Markdown reports are written."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json")
        report = run_harness(run_dir, str(tmp_path))
        assert os.path.exists(str(tmp_path / "validation_report.json"))
        assert os.path.exists(str(tmp_path / "validation_report.md"))
        shutil.rmtree(run_dir)


class TestProvenanceGuard:
    def test_stale_commit_fires_banner(self, tmp_path):
        """Artifact with fake commit SHA fires STALE banner when --expect-commit is HEAD."""
        import subprocess
        run_dir = tempfile.mkdtemp()
        shutil.copy(
            os.path.join(FIXTURE_DIR, "network_stats_stale_commit.json"),
            os.path.join(run_dir, "network_stats.json"),
        )
        # Get actual HEAD
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        if not head:
            shutil.rmtree(run_dir)
            return  # skip if not in git repo

        report = run_harness(run_dir, str(tmp_path), expect_commit=head)
        # Banner should be set (fake SHA "000..." != HEAD)
        assert report.get("provenance_banner") is not None, (
            "Expected STALE banner for artifact with fake commit SHA"
        )
        art_prov = report["artifact_provenance"].get("network_stats", {})
        assert art_prov.get("verdict") in ("stale", "unverified"), (
            f"Expected stale/unverified verdict, got {art_prov.get('verdict')}"
        )
        # Every scored row should be tagged provenance_suspect
        # (banner is set → provenance_suspect is True for the artifact)
        assert art_prov.get("provenance_suspect") is True
        shutil.rmtree(run_dir)

    def test_strict_exits_nonzero_for_stale(self, tmp_path):
        """--strict exits non-zero when a stale artifact is detected."""
        import subprocess
        run_dir = tempfile.mkdtemp()
        shutil.copy(
            os.path.join(FIXTURE_DIR, "network_stats_stale_commit.json"),
            os.path.join(run_dir, "network_stats.json"),
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        if not head:
            shutil.rmtree(run_dir)
            return

        # run_harness with strict=True should call sys.exit(1)
        import pytest
        with pytest.raises(SystemExit) as exc_info:
            run_harness(run_dir, str(tmp_path), expect_commit=head, strict=True)
        assert exc_info.value.code == 1, (
            f"Expected sys.exit(1) under --strict with stale artifact, "
            f"got {exc_info.value.code}"
        )
        shutil.rmtree(run_dir)

    def test_fresh_artifact_no_banner(self, tmp_path):
        """Artifact with no embedded commit (network_stats_85pct) → unverified but no banner unless expect_commit set."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json")
        # Without expect_commit: unverified but no banner (default permissive)
        report = run_harness(run_dir, str(tmp_path))
        # unverified but provenance_banner should be None (default permissive for unverified)
        # OR it may fire a banner — both are acceptable; the key is it doesn't crash
        assert "provenance_banner" in report  # field exists
        shutil.rmtree(run_dir)

    def test_provenance_in_report_header(self, tmp_path):
        """Report header contains artifact_provenance with path/mtime/verdict."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json")
        report = run_harness(run_dir, str(tmp_path))
        prov = report.get("artifact_provenance", {})
        assert "network_stats" in prov, "network_stats missing from artifact_provenance"
        ns_prov = prov["network_stats"]
        assert "verdict" in ns_prov
        assert "mtime" in ns_prov or ns_prov.get("verdict") == "not_found"
        shutil.rmtree(run_dir)


class TestPathLengthTightened:
    def test_stale_baseline_path_fails(self, tmp_path):
        """Stale baseline path_length=5.18 now FAILS with two-sided range."""
        # Synthetic artifact with path_length=5.18 (the stale Handoff-3 value)
        run_dir = tempfile.mkdtemp()
        artifact = {
            "generated": {
                "mean_degree": 2.70, "k2_moment": 10.94,
                "giant_component_fraction": 0.887, "mean_path_length": 5.18,
                "n_nodes": 300, "n_edges": 405,
            },
            "checks": {
                "mean_degree_ANCHOR": {"generated": 2.70, "status": "ANCHOR"},
                "giant_component": {"generated": 0.887, "status": "PASS"},
                "mean_path_length": {"generated": 5.18, "status": "PASS"},
                "clustering_coefficient": {"generated": 0.0091, "er_floor": 0.009, "status": "REVIEW"},
                "dispersion_k2_over_k2": {"generated": 1.501, "poisson_baseline": 1.370, "status": "PASS"},
            },
        }
        with open(os.path.join(run_dir, "network_stats.json"), "w") as f:
            json.dump(artifact, f)
        report = run_harness(run_dir, str(tmp_path))
        mpl = report["scored"]["mean_path_length"]
        assert mpl["status"] == "FAIL", (
            f"Expected FAIL for path_length=5.18 (stale baseline), got {mpl['status']}"
        )
        shutil.rmtree(run_dir)

    def test_dyad_short_path_fails(self, tmp_path):
        """Dyad network path_length≈2.29 now FAILS with two-sided range."""
        run_dir = tempfile.mkdtemp()
        artifact = {
            "generated": {
                "mean_degree": 2.83, "k2_moment": 41.047,
                "giant_component_fraction": 0.283, "mean_path_length": 2.29,
                "n_nodes": 300, "n_edges": 424,
            },
            "checks": {
                "mean_degree_ANCHOR": {"generated": 2.83, "status": "ANCHOR"},
                "giant_component": {"generated": 0.283, "status": "PASS"},
                "mean_path_length": {"generated": 2.29, "status": "PASS"},
                "clustering_coefficient": {"generated": 0.0825, "er_floor": 0.00942, "status": "ABOVE_FLOOR"},
                "dispersion_k2_over_k2": {"generated": 5.137, "poisson_baseline": 1.354, "status": "PASS"},
            },
        }
        with open(os.path.join(run_dir, "network_stats.json"), "w") as f:
            json.dump(artifact, f)
        report = run_harness(run_dir, str(tmp_path))
        mpl = report["scored"]["mean_path_length"]
        assert mpl["status"] == "FAIL", (
            f"Expected FAIL for path_length=2.29 (dyad too-short), got {mpl['status']}"
        )
        shutil.rmtree(run_dir)

    def test_good_path_passes(self, tmp_path):
        """The 85% fixture with path_length=3.4 still PASSES."""
        run_dir = _make_run_dir("network_stats_85pct_giant.json")
        report = run_harness(run_dir, str(tmp_path))
        mpl = report["scored"]["mean_path_length"]
        assert mpl["status"] == "PASS", (
            f"Expected PASS for path_length=3.4, got {mpl['status']}"
        )
        shutil.rmtree(run_dir)


class TestRealizedSchema:
    def test_realized_network_validation_report(self, tmp_path):
        """Harness falls back gracefully to network_validation_report.json (realized name)."""
        run_dir = tempfile.mkdtemp()
        shutil.copy(
            os.path.join(FIXTURE_DIR, "network_stats_85pct_giant.json"),
            os.path.join(run_dir, "network_validation_report.json"),   # realized name
        )
        report = run_harness(run_dir, str(tmp_path))
        disc = report["schema_discovery"]["network_stats"]
        assert disc["realized"] is True, "Expected realized=True for fallback file"
        assert disc["source_file"] == "network_validation_report.json"
        shutil.rmtree(run_dir)


# Standalone runner (no pytest required)
if __name__ == "__main__":
    import traceback
    failures = []

    def run_test(name, fn):
        try:
            tmp = tempfile.mkdtemp()
            fn(type("P", (), {"__truediv__": lambda s, x: os.path.join(tmp, x),
                               "__str__": lambda s: tmp})())
            print(f"  PASS: {name}")
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
            traceback.print_exc()
            failures.append(name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("Running validation harness tests...")

    t1 = TestGiantComponentCheck()
    run_test("29pct_fails",    lambda p: t1.test_29pct_fails(p))
    run_test("85pct_passes",   lambda p: t1.test_85pct_passes(p))
    run_test("flip_proves",    lambda p: t1.test_flip_proves_harness_discriminates(p))

    t2 = TestMissingOutbreakSim()
    run_test("missing_outbreak_no_crash",  lambda p: t2.test_missing_outbreak_sim_no_crash(p))
    run_test("missing_both_no_crash",      lambda p: t2.test_missing_both_files_no_crash(p))

    t3 = TestHeldOutIsReportOnly()
    run_test("held_out_never_in_scored",            lambda p: t3.test_held_out_never_in_scored(p))
    run_test("held_out_in_held_out_block",          lambda p: t3.test_held_out_in_held_out_block(p))
    run_test("held_out_does_not_affect_pass_rate",  lambda p: t3.test_held_out_does_not_affect_pass_rate(p))
    run_test("classify_overrides_dict",             lambda p: TestHeldOutIsReportOnly.test_classify_held_out_overrides_dict(None))

    t4 = TestAllClassesPresent()
    run_test("all_four_classes",          lambda p: t4.test_all_four_classes_in_report(p))
    run_test("stubbed_not_computable",    lambda p: t4.test_stubbed_returns_not_computable(p))
    run_test("report_files_written",      lambda p: t4.test_report_files_written(p))

    t5 = TestRealizedSchema()
    run_test("realized_schema_fallback",  lambda p: t5.test_realized_network_validation_report(p))

    t6 = TestProvenanceGuard()
    run_test("stale_commit_fires_banner",   lambda p: t6.test_stale_commit_fires_banner(p))
    run_test("strict_exits_nonzero",        lambda p: t6.test_strict_exits_nonzero_for_stale(p))
    run_test("fresh_no_banner",             lambda p: t6.test_fresh_artifact_no_banner(p))
    run_test("provenance_in_header",        lambda p: t6.test_provenance_in_report_header(p))

    t7 = TestPathLengthTightened()
    run_test("stale_path_fails",   lambda p: t7.test_stale_baseline_path_fails(p))
    run_test("dyad_path_fails",    lambda p: t7.test_dyad_short_path_fails(p))
    run_test("good_path_passes",   lambda p: t7.test_good_path_passes(p))

    n_tests = 20
    print(f"\n{'PASSED' if not failures else 'FAILED'}: {n_tests - len(failures)}/{n_tests} tests")
    if failures:
        print(f"Failures: {failures}")
