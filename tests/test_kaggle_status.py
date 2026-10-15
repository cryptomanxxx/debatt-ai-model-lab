"""Regression tests for Kaggle CLI worker-status parsing."""
import re
from pathlib import Path

SCRIPT = Path("scripts/run_kaggle_kernel.sh").read_text(encoding="utf-8")


def test_completion_requires_explicit_worker_field():
    assert 'KernelWorkerStatus.COMPLETE)' in SCRIPT
    assert "grep -Eo 'KernelWorkerStatus\\.[A-Z_]+'" in SCRIPT
    assert "'(^|[^a-z])complete([^a-z]|$)'" not in SCRIPT


def test_username_cannot_fake_completion():
    status = 'complete-lab/debatt-ai-smollm3-gpu has status "KernelWorkerStatus.QUEUED"'
    match = re.search(r"KernelWorkerStatus\.[A-Z_]+", status)
    assert match is not None
    assert match.group() == "KernelWorkerStatus.QUEUED"


def test_cancellation_not_treated_as_complete():
    for name in ("CANCELED", "CANCELLED", "CANCEL_ACKNOWLEDGED"):
        assert f"KernelWorkerStatus.{name}" in SCRIPT
        assert name != "COMPLETE"
