"""Do not mistake infrastructure failures for a reproduced software bug."""
from pathlib import Path
import pytest
from evaluation import harness, models


@pytest.mark.parametrize("source", [
    'raise ImportError("missing dependency")',
    'import pytest\n@pytest.fixture\ndef unavailable():\n    raise RuntimeError("setup failed")\ndef test_fixture(unavailable):\n    pass',
    '# no tests collected',
])
def test_pytest_execution_errors_are_not_assertion_failures(tmp_path, source):
    (tmp_path / "test_probe.py").write_text(source)
    with pytest.raises(harness.HarnessError, match="valid test run"):
        harness._run_pytest(tmp_path, ".")


@pytest.mark.parametrize("condition,passed,failed", [("True", True, 0), ("False", False, 1)])
def test_real_assertions_produce_valid_outcomes(tmp_path, condition, passed, failed):
    (tmp_path / "test_probe.py").write_text(f"def test_probe():\n    assert {condition}\n")
    result = harness._run_pytest(tmp_path, ".")
    assert result.passed is passed
    assert result.n_failed == failed


def test_evaluation_rejects_broken_issue_test_import(monkeypatch):
    prepare = harness._prepare
    def with_broken_import(spec, workdir):
        prepare(spec, workdir)
        (workdir / "case_tests" / "test_import_error.py").write_text('raise ImportError("missing dependency")')
    monkeypatch.setattr(harness, "_prepare", with_broken_import)
    spec = models.discover(harness.REPO_ROOT / "cases")[0]
    with pytest.raises(harness.HarnessError, match="valid test run"):
        harness.evaluate(spec)
