"""Summarize audited Tahoe eligibility overlaps by historical evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad

import epi_c3_tahoe_isolation_v1 as isolation


def ids_from_obs(path: Path, legacy: bool) -> set[str]:
    dataset = ad.read_h5ad(path, backed="r")
    try:
        if legacy:
            ids = set()
            for value in dataset.obs["source_cell_id"]:
                parts = value.split("|")
                if len(parts) != 2:
                    raise ValueError("Invalid legacy Tahoe evaluation identity")
                ids.add(isolation.identity(*parts))
        else:
            ids = set(map(str, dataset.obs_names))
        if len(ids) != dataset.n_obs:
            raise ValueError("A Tahoe evaluation split contains duplicate source identities")
        return ids
    finally:
        dataset.file.close()


def run(run_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    protected_dir = run_root / "tahoe-protected-identities-v1"
    audit_dir = run_root / "tahoe-foundation-isolation-v1"
    manifest_path = protected_dir / "manifest.json"
    audit_path = audit_dir / "audit.json"
    verification_path = run_root / "tahoe-foundation-isolation-verification-v1/verification.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    progress_path = audit_dir / "progress.private.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if (
        audit.get("state") != "complete" or progress.get("state") != "complete"
        or verification.get("state") != "complete" or not verification.get("per_identity_counts_match")
        or verification["audit_sha256"] != isolation.sha256(audit_path)
        or audit["protected_manifest_sha256"] != isolation.sha256(manifest_path)
        or manifest["private_sha256"] != isolation.sha256(protected_dir / manifest["private_file"])
        or verification["protected_eligible_rows"] != audit["protected_eligible_rows"]
    ):
        raise ValueError("Tahoe isolation verification is missing, incomplete, or stale")
    eligible = set(progress["protected_eligible"])
    if sum(progress["protected_eligible"].values()) != audit["protected_eligible_rows"]:
        raise ValueError("Private Tahoe eligible counts differ from the frozen audit")
    rows, evaluated = [], set()
    for artifact in manifest["evaluation_artifacts"]:
        label = artifact["label"]
        name, role = label.split("/")
        legacy = name.startswith("tahoe-demo-")
        if legacy:
            path = run_root.parent / name / "split" / f"{role}.h5ad"
        else:
            directory = "tahoe-interim-eval-v1" if name == "interim" else "tahoe-real-evaluation-smoke-v1"
            path = run_root / directory / f"{role}.h5ad"
        if isolation.sha256(path) != artifact["sha256"]:
            raise ValueError("A historical Tahoe evaluation split changed")
        ids = ids_from_obs(path, legacy)
        if len(ids) != artifact["identities"]:
            raise ValueError("A historical Tahoe evaluation split changed size")
        evaluated.update(ids)
        rows.append({
            "evaluation": label, "unique_identities": len(ids),
            "eligible_rows_under_old_filter": sum(progress["protected_eligible"].get(cell_id, 0) for cell_id in ids),
        })
    protected = {item["cell_id"] for item in json.loads(
        (protected_dir / manifest["private_file"]).read_text(encoding="utf-8")
    )["records"]}
    if evaluated != protected or not eligible.issubset(protected):
        raise ValueError("The evaluation splits differ from the protected identity inventory")
    summary = {
        "format": "tahoe-foundation-overlap-summary-v1", "state": "complete",
        "audit_sha256": isolation.sha256(audit_path),
        "verification_sha256": isolation.sha256(verification_path),
        "progress_private_sha256": isolation.sha256(progress_path),
        "script_sha256": isolation.sha256(Path(__file__)),
        "unique_eligible_protected_identities": len(eligible),
        "eligible_protected_rows": audit["protected_eligible_rows"],
        "evaluations": rows,
        "notes": [
            "Evaluation splits may overlap, so their counts must not be added.",
            "Eligibility under the old filter does not prove that a particular training snapshot already consumed a row.",
            "This assessment covers explicitly inventoried artifacts, not every possible benchmark or cross-dataset duplicate.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"eligible_protected_rows": summary["eligible_protected_rows"], "evaluations": rows}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_root, args.output)
