"""Smoke test: the synthetic demo runs end-to-end and produces a non-empty figure."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "eigen_spectrum.py"


def test_synthetic_demo(tmp_path: Path) -> None:
    out = tmp_path / "figures"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr

    fig = out / "eigen_spectrum.png"
    assert fig.exists(), "demo figure not written"
    assert fig.stat().st_size > 5_000, "demo figure suspiciously small"

    payload = json.loads((out.parent / "rmt_eigen.json").read_text())
    assert payload["asset"] == "synthetic"
    assert payload["lambda_plus"] > 0
    assert len(payload["eigs"]) > 0


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_synthetic_demo(Path(d))
        print("ok")
