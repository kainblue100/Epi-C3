"""Verify an existing public ENCODE ATAC peak source without exposing sample IDs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse

import download_encode_peaks as peaks
import freeze_encode_catalogs as freeze
import select_encode_prior as selection


ATAC_ACCESSION = "ENCFF855PCP"
ATAC_EXPERIMENT = "ENCSR868FGK"
ATAC_BYTES = 4047527
ATAC_MD5 = "bee81b0efe50b144c24d2060a87aa843"
ATAC_SHA = "47a71eaac0bfbad586a14d89971398c8922bf8864b5eeab4cb58d37ae925e0e9"
ATAC_METADATA_SHA = "c5fae8ecec0918a10d0cf4df1d971f683a8c8fbe3cb163447cea6a684ef468e1"
FIELDS = tuple(field for field in freeze.EXPERIMENT_FIELDS if not field.startswith("target."))


def experiment_url() -> str:
    params = [("type", "Experiment"), ("accession", ATAC_EXPERIMENT), ("format", "json"), ("limit", "all")]
    params.extend(("field", field) for field in FIELDS)
    return freeze.HOST + "/search/?" + urllib.parse.urlencode(params)


def verify_metadata(file_metadata: dict, experiment: dict) -> dict:
    replicates = file_metadata.get("biological_replicates")
    if (
        file_metadata.get("accession") != ATAC_ACCESSION
        or file_metadata.get("dataset") != f"/experiments/{ATAC_EXPERIMENT}/"
        or file_metadata.get("status") != "released" or file_metadata.get("assembly") != "GRCh38"
        or file_metadata.get("file_format") != "bed" or file_metadata.get("file_format_type") != "narrowPeak"
        or file_metadata.get("output_type") != "conservative IDR thresholded peaks"
        or file_metadata.get("file_size") != ATAC_BYTES or file_metadata.get("md5sum") != ATAC_MD5
        or file_metadata.get("href") != f"/files/{ATAC_ACCESSION}/@@download/{ATAC_ACCESSION}.bed.gz"
    ):
        raise ValueError("ATAC file metadata differs from the frozen public source")
    if (
        experiment.get("accession") != ATAC_EXPERIMENT or experiment.get("status") != "released"
        or experiment.get("assay_title") != "ATAC-seq"
        or experiment.get("biosample_ontology", {}).get("term_name") != "K562"
        or experiment.get("biosample_summary") != "Homo sapiens K562"
        or experiment.get("perturbed") is not False
        or not isinstance(experiment.get("lab", {}).get("title"), str)
        or not experiment["lab"]["title"]
    ):
        raise ValueError("ATAC experiment is not the expected released untreated K562 source")
    reason = selection.unmodified_replicates(experiment, replicates)
    if reason is not None:
        raise ValueError(f"ATAC replicate provenance is not clean: {reason}")
    return {
        "experiment_accession": ATAC_EXPERIMENT,
        "file_accession": ATAC_ACCESSION,
        "producing_lab": experiment["lab"]["title"],
        "assembly": "GRCh38",
        "output_type": file_metadata["output_type"],
        "biological_replicates": sorted(replicates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak", type=Path, required=True)
    parser.add_argument("--file-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = selection.common.read_json(args.file_metadata, ATAC_METADATA_SHA)
    data, count = freeze.fetch_catalog(experiment_url(), "ATAC experiment")
    if count != 1:
        raise ValueError("The exact ATAC experiment query must return one record")
    experiment = json.loads(data)["@graph"][0]
    source = verify_metadata(metadata, experiment)
    verified = peaks.verify_peak_file(args.peak, ATAC_BYTES, ATAC_MD5, ATAC_SHA)
    if shutil.disk_usage(args.output_dir.parent).free < selection.common.RESERVE_WITH_HEADROOM:
        raise OSError("Insufficient disk space above the 200 GiB reserve and 1 GiB headroom")
    args.output_dir.mkdir(mode=0o700)
    freeze.write_new(args.output_dir / "experiment-metadata.json", data)
    report = {
        "format": "epi-c3-encode-k562-atac-verified-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_query": experiment_url(),
        "input_sha256": {
            "experiment_metadata": hashlib.sha256(data).hexdigest(),
            "file_metadata": ATAC_METADATA_SHA,
            "verifier_script": selection.common.sha256(Path(__file__)),
            "peak_validator_script": selection.common.sha256(Path(peaks.__file__)),
            "selector_script": selection.common.sha256(Path(selection.__file__)),
        },
        "source": {**source, **verified},
        "limitations": [
            "K562 ATAC and K562 ChIP are unpaired bulk assays; neither is paired with benchmark cells.",
            "The public metadata identify no treatment or genetic modification, but independence of laboratory cultures is not established.",
            "ATAC accessibility and ChIP binding do not prove causal regulation.",
        ],
    }
    digest = selection.write_json(args.output_dir / "atac-source.json", report)
    print(json.dumps({"state": "complete", "report_sha256": digest, "rows": verified["rows"],
                      "invalid_summit_rows_excluded": verified["invalid_summit_rows_excluded"]}), flush=True)


if __name__ == "__main__":
    main()
