#!/usr/bin/env python3
"""Fail-fast validation for the OpenArm IEEE Access research bundle."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

try:
    from pypdf import PdfReader
except ModuleNotFoundError:
    PdfReader = None


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "README.md",
    ROOT / "docs" / "RESEARCH_AUDIT_KO.md",
    ROOT / "docs" / "EXPERIMENT_PROTOCOL_KO.md",
    ROOT / "docs" / "VENUE_AND_CLAIM_BOUNDARIES_KO.md",
    ROOT / "references" / "references.bib",
    ROOT / "references" / "SOURCES.md",
    ROOT / "references" / "SHA256SUMS",
    ROOT / "source_files" / "hardware" / "ROMO-B_manual_complete_compressed.pdf",
    ROOT / "source_files" / "hardware" / "ROMO-B_manual_verified_complete.md",
    ROOT / "simulation" / "config" / "romo_b_openarm_spec.json",
    ROOT / "simulation" / "assets" / "romo_b_openarm_manual_proxy.usda",
    ROOT / "simulation" / "README.md",
    ROOT / "simulation" / "scripts" / "build_romo_b_openarm_usda.py",
    ROOT / "simulation" / "scripts" / "run_isaacsim_smoke.py",
    ROOT / "simulation" / "scripts" / "run_usd_tool.sh",
    ROOT / "simulation" / "scripts" / "validate_isaacsim_model.py",
    ROOT / "simulation" / "tests" / "test_model_contract.py",
    ROOT / "docs" / "ISAAC_SIM_ROMOB_OPENARM_KO.md",
]
DOIS = [
    "10.1017/S0263574798000873",
    "10.1109/ROBIO.2011.6181521",
    "10.1115/1.4037287",
    "10.1109/TITS.2024.3432885",
    "10.1115/1.4033841",
    "10.1109/ACCESS.2019.2915115",
]


def validate_checksums(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = re.split(r"\s+", line, maxsplit=1)
        relative = relative.lstrip("*")
        target = ROOT / relative
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Checksum mismatch: {relative}")


def pdf_page_count(path: Path) -> int:
    if PdfReader is not None:
        return len(PdfReader(str(path)).pages)
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Install pypdf or Poppler pdfinfo to validate the manual page count") from error
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not read page count from pdfinfo output for {path}")
    return int(match.group(1))


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")

    oversized = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size >= 100_000_000
    ]
    if oversized:
        raise RuntimeError(f"Files exceed GitHub's 100 MB limit: {oversized}")

    manual = ROOT / "source_files" / "hardware" / "ROMO-B_manual_complete_compressed.pdf"
    page_count = pdf_page_count(manual)
    if page_count != 38:
        raise RuntimeError(f"Compressed ROMO-B manual has {page_count} pages, expected 38")

    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "docs" / "RESEARCH_AUDIT_KO.md",
            ROOT / "references" / "references.bib",
            ROOT / "references" / "SOURCES.md",
        ]
    ).lower()
    missing_dois = [doi for doi in DOIS if doi.lower() not in corpus]
    if missing_dois:
        raise RuntimeError(f"Required DOI references are missing: {missing_dois}")

    validate_checksums(ROOT / "references" / "SHA256SUMS")
    print("bundle validation: PASS")
    print(f"files: {sum(1 for path in ROOT.rglob('*') if path.is_file())}")
    print(f"compressed manual pages: {page_count}")


if __name__ == "__main__":
    main()
