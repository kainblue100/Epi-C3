import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from c3vcell.calibration import OBJECTIVE, calibrate_effect
from c3vcell.cli import build_parser
from c3vcell.data import CONTROL, write_split
from c3vcell.ensemble import _check_members
from c3vcell.epigenetics import (
    CONDITION, COORDINATES, FEATURES, NETWORK_FEATURES, _shuffle_graph_targets, build_epigenetic_bundle,
    load_epigenetic_bundle, shuffle_epigenetic_bundle,
)
from c3vcell.integrity import file_sha256
from c3vcell.mean_response import fit_mean_response, predict_mean_response
from c3vcell.model import C3VCell, ModelConfig
from c3vcell.predict import predict_holdout, predict_targets
from c3vcell.research_review import review_network_diagnostic
from c3vcell.scoring import SCORED_METRICS
from c3vcell.shared_calibration import calibrate_shared_residual
from c3vcell.training import load_model, train


@pytest.fixture
def epi_inputs(tmp_path: Path) -> dict[str, Path]:
    genes = ["SP1", "USF1", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "AMBIG"]
    ids = [f"ENSG{i:011d}" for i in range(1, len(genes) + 1)]
    ids[-1] += "|ENSG99999999999"
    targets = genes[:8]
    labels = [CONTROL] * 60 + sum(([target] * 8 for target in targets), [])
    rng = np.random.default_rng(42)
    data = ad.AnnData(
        sp.csr_matrix(rng.poisson(3, size=(len(labels), len(genes))).astype(np.int32) + 1),
        obs=pd.DataFrame(
            {"target_gene": labels, "context": "HCT116", "batch": "b"},
            index=[f"cell-{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame({"ensembl_ids": ids}, index=genes),
        uns={"c3vcell_genome_assembly": "GRCh38"},
    )
    split = tmp_path / "outer"
    write_split(
        data, split, control=CONTROL, holdout_fraction=0.25, seed=4,
        holdout_targets=["G5", "G6"],
    )
    gtf = tmp_path / "genes.gtf"
    with gtf.open("w", encoding="utf-8") as stream:
        for index, ensembl in enumerate(ids[:-1]):
            tss = 10_000 + index * 10_000
            stream.write(
                f'chr1\ttest\tgene\t{tss + 1}\t{tss + 101}\t.\t+\t.\t'
                f'gene_id "{ensembl}.1"; gene_name "{genes[index]}";\n'
            )
    atac = tmp_path / "atac.bed"
    atac.write_text("chr1\t30000\t30020\n", encoding="utf-8")
    sp1 = tmp_path / "sp1.bed"
    sp1.write_text("chr1\t30005\t30015\n", encoding="utf-8")
    ctcf = tmp_path / "ctcf.bed"
    ctcf.write_text("chr1\t40000\t40020\n", encoding="utf-8")
    loops = tmp_path / "loops.bedpe"
    loops.write_text("chr1\t30000\t30020\tchr1\t40000\t40020\n", encoding="utf-8")

    def source(role: str, path: Path, **metadata: object) -> dict:
        return {
            "role": role, "path": str(path), "sha256": file_sha256(path),
            "source_url": f"https://example.org/{path.name}", "assembly": "GRCh38", **metadata,
        }

    common = {"context": "HCT116", "condition": CONDITION, "coordinates": COORDINATES}
    spec = {
        "format_version": 1, "context": "HCT116", "assembly": "GRCh38",
        "condition": CONDITION,
        "sources": [
            source("annotation", gtf),
            source("atac", atac, **common, assay="ATAC-seq", target=None, accession="ATAC1"),
            source("sp1", sp1, **common, assay="TF ChIP-seq", target="SP1", accession="SP1-1"),
            source("ctcf", ctcf, **common, assay="TF ChIP-seq", target="CTCF", accession="CTCF1"),
            source(
                "hic_loops", loops, **common, assay="Hi-C", resolution_bp=10_000,
                covered_chromosomes=["chr1"], evidence_type="called-loop",
            ),
        ],
    }
    spec_path = tmp_path / "sources.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    bundle = tmp_path / "bundle"
    build_epigenetic_bundle(split / "train.h5ad", spec_path, bundle, promoter_flank=50)
    return {
        "source": split / "train.h5ad", "reference": split / "reference.h5ad",
        "controls": split / "input_controls.h5ad", "bundle": bundle, "spec": spec_path,
    }


def _train(inputs: dict[str, Path], outdir: Path, seed: int = 3) -> Path:
    return train(
        inputs["source"], outdir, latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=3, batch_size=4, seed=seed,
        epigenetic_bundle=inputs["bundle"],
    )


def test_target_inference_enforces_epigenome_context_and_gene_axis(
    epi_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    checkpoint = _train(epi_inputs, tmp_path / "target-model")
    requests = tmp_path / "target-requests.csv"
    requests.write_text("context,target_gene\nHCT116,G5\nHCT116,G6\n", encoding="utf-8")
    output = tmp_path / "target-prediction.h5ad"
    predict_targets(
        checkpoint, requests, epi_inputs["controls"], output,
        cells_per_target=4, steps=2, sampling="residual-rounding", effect_strength=0,
    )
    prediction = ad.read_h5ad(output)
    assert prediction.shape == (8, 10)
    assert set(prediction.obs["target_gene"]) == {"G5", "G6"}
    wrong = ad.read_h5ad(epi_inputs["controls"])
    wrong.uns["c3vcell_genome_assembly"] = "GRCh37"
    wrong_path = tmp_path / "wrong-assembly.h5ad"
    wrong.write_h5ad(wrong_path)
    with pytest.raises(ValueError, match="assembly"):
        predict_targets(
            checkpoint, requests, wrong_path, tmp_path / "invalid-assembly-prediction.h5ad",
        )


@pytest.fixture
def network_inputs(epi_inputs: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    methylation = tmp_path / "cpg.bed"
    methylation.write_text(
        "#chrom\tstart\tend\tfraction\tcoverage\n"
        "chr1\t30006\t30007\t0.25\t20\n"
        "chr1\t30008\t30009\t0.75\t20\n"
        "chr1\t40002\t40003\t0.8\t10\n"
        "chr1\t50002\t50003\t0\t5\n",
        encoding="utf-8",
    )
    spec = json.loads(epi_inputs["spec"].read_text(encoding="utf-8"))
    spec["format_version"] = 2
    contacts = spec["sources"][-1]
    contacts.update(
        role="chromatin_contacts", assay="ChIA-PET", target="RAD21",
        resolution_bp=None, resolution_kind="anchor-interval",
    )
    spec["sources"].append({
        "role": "methylation", "path": str(methylation), "sha256": file_sha256(methylation),
        "source_url": "https://example.org/cpg.bed", "assembly": "GRCh38",
        "context": "HCT116", "condition": CONDITION, "coordinates": COORDINATES,
        "assay": "RRBS", "file_format": "cpg-fraction-bed", "min_coverage": 5,
    })
    spec_path = tmp_path / "network-sources.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    bundle = tmp_path / "network-bundle"
    build_epigenetic_bundle(epi_inputs["source"], spec_path, bundle, promoter_flank=50)
    return {**epi_inputs, "bundle": bundle, "spec": spec_path, "methylation": methylation}


@pytest.fixture
def uncovered_inputs(network_inputs: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    methylation = tmp_path / "reference-cpg-fractions.bed"
    methylation.write_text(
        "#chrom\tstart\tend\tfraction\n"
        "chr1\t30006\t30007\t0.25\n"
        "chr1\t30008\t30009\t0.75\n"
        "chr1\t40002\t40003\t0.8\n"
        "chr1\t50002\t50003\t0\n"
        "chr1\t999999\t1000000\t0.5\n",
        encoding="utf-8",
    )
    loops = tmp_path / "multi-resolution.bedpe"
    loops.write_text(
        "chr1\t30000\t30020\tchr1\t40000\t40020\n"
        "chr1\t70000\t70040\tchr1\t60000\t60040\n",
        encoding="utf-8",
    )
    spec = json.loads(network_inputs["spec"].read_text(encoding="utf-8"))
    contacts = next(source for source in spec["sources"] if source["role"] == "chromatin_contacts")
    contacts.update(
        path=str(loops), sha256=file_sha256(loops), assay="Hi-C", target=None,
        resolution_bp=None, resolution_kind="multi-resolution-anchor-interval",
        anchor_widths_bp=[20, 40],
    )
    methyl = spec["sources"][-1]
    methyl.update(
        path=str(methylation), sha256=file_sha256(methylation), assay="WGBS",
        file_format="cpg-fraction-bed-no-coverage", min_coverage=None,
        site_coverage="not-available", aggregation="equal-strand-then-cpg-mean",
        reference_sequence_validated=True, covered_chromosomes=["chr1"],
    )
    spec_path = tmp_path / "uncovered-sources.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return {**network_inputs, "spec": spec_path, "methylation": methylation, "loops": loops}


def test_uncovered_wgbs_uses_equal_cpg_means_and_multi_resolution_loops(
    uncovered_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    bundle = tmp_path / "uncovered-bundle"
    build_epigenetic_bundle(uncovered_inputs["source"], uncovered_inputs["spec"], bundle, promoter_flank=50)
    data = ad.read_h5ad(uncovered_inputs["source"])
    values, masks, metadata = load_epigenetic_bundle(bundle, data)
    gene = {name: index for index, name in enumerate(data.var_names)}
    promoter = NETWORK_FEATURES.index("promoter_methylation")
    linked = NETWORK_FEATURES.index("linked_element_methylation")
    assert values[gene["G1"], promoter] == pytest.approx(0.5)
    assert values[gene["G2"], promoter] == pytest.approx(0.8)
    assert values[gene["G2"], linked] == pytest.approx(0.5)
    assert values[gene["G3"], promoter] == 0 and masks[gene["G3"], promoter] == 1
    assert values[gene["G4"], promoter] == 0 and masks[gene["G4"], promoter] == 0
    assert metadata["loop_count"] == 2
    assert "read coverage is unavailable" in metadata["semantics"]["methylation"]
    assert "reference-sequence-validated CpG dyads" in metadata["semantics"]["methylation"]


@pytest.mark.parametrize("update,missing", [
    ({"min_coverage": 5}, None),
    ({"site_coverage": "available"}, None),
    ({"aggregation": "coverage-weighted"}, None),
    ({"reference_sequence_validated": False}, None),
    ({"covered_chromosomes": []}, None),
    ({"covered_chromosomes": ["chr1", "chr1"]}, None),
    ({}, "min_coverage"),
])
def test_uncovered_wgbs_requires_explicit_honest_metadata(
    uncovered_inputs: dict[str, Path], tmp_path: Path, update: dict, missing: str | None,
) -> None:
    spec = json.loads(uncovered_inputs["spec"].read_text(encoding="utf-8"))
    source = spec["sources"][-1]
    source.update(update)
    if missing:
        source.pop(missing)
    path = tmp_path / "invalid-wgbs-spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="Methylation needs"):
        build_epigenetic_bundle(uncovered_inputs["source"], path, tmp_path / "invalid-wgbs")


@pytest.mark.parametrize("line,match", [
    ("chr1\t30006\t30007\t1.1\n", "invalid CpG"),
    ("chr1\t30006\t30007\tnan\n", "invalid CpG"),
    ("chr1\t30006\t30008\t0.5\n", "invalid CpG"),
    ("chr1\t30006\t30007\t0.5\nchr1\t30006\t30007\t0.2\n", "duplicated or unsorted"),
    ("chr1\t40000\t40001\t0.5\nchr1\t30006\t30007\t0.2\n", "duplicated or unsorted"),
    ("chr2\t30006\t30007\t0.5\n", "invalid CpG"),
    ("chr1\t999999\t1000000\t0.5\n", "No observed CpGs"),
])
def test_uncovered_wgbs_rejects_invalid_fractions(
    uncovered_inputs: dict[str, Path], tmp_path: Path, line: str, match: str,
) -> None:
    uncovered_inputs["methylation"].write_text(line, encoding="utf-8")
    spec = json.loads(uncovered_inputs["spec"].read_text(encoding="utf-8"))
    spec["sources"][-1]["sha256"] = file_sha256(uncovered_inputs["methylation"])
    path = tmp_path / "invalid-fractions.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        build_epigenetic_bundle(uncovered_inputs["source"], path, tmp_path / "invalid-fractions")


@pytest.mark.parametrize("widths,resolution,match", [
    ([], None, "metadata"),
    ([20, 20], None, "metadata"),
    ([40, 20], None, "metadata"),
    ([20, 0], None, "metadata"),
    ([20, True], None, "metadata"),
    ([20, 40], 20, "metadata"),
    ([20], None, "undeclared interval width"),
    ([20, 40, 60], None, "do not match the loops"),
])
def test_multi_resolution_loop_metadata_matches_observed_anchors(
    uncovered_inputs: dict[str, Path], tmp_path: Path, widths: list, resolution: int | None, match: str,
) -> None:
    spec = json.loads(uncovered_inputs["spec"].read_text(encoding="utf-8"))
    contacts = next(source for source in spec["sources"] if source["role"] == "chromatin_contacts")
    contacts.update(anchor_widths_bp=widths, resolution_bp=resolution)
    path = tmp_path / "invalid-widths.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        build_epigenetic_bundle(uncovered_inputs["source"], path, tmp_path / "invalid-widths")


def test_exploratory_research_review_verifies_artifacts_and_blocks_validation_claims(
    network_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    parent, previous, run = tmp_path / "parent", tmp_path / "previous", tmp_path / "review-run"
    (parent / "split").mkdir(parents=True)
    previous.mkdir()
    run.mkdir()
    for label, name in (("source", "train.h5ad"), ("reference", "reference.h5ad"),
                        ("controls", "input_controls.h5ad")):
        shutil.copyfile(network_inputs[label], parent / "split" / name)
    data = ad.read_h5ad(parent / "split" / "train.h5ad")
    reference = ad.read_h5ad(parent / "split" / "reference.h5ad")
    controls = ad.read_h5ad(parent / "split" / "input_controls.h5ad")
    training = sorted(set(data.obs["target_gene"].astype(str)) - {CONTROL})
    heldout = sorted(set(reference.obs["target_gene"].astype(str)) - {CONTROL})

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def metric_values(mse: float) -> dict:
        return {"HCT116": {
            "pds_cosine": 0.5, "expr_mse_unbiased_capped_norm": mse,
            "de_wilcoxon_direction_fidelity_yield_raw": 0.5,
            "de_wilcoxon_direction_reach_raw": None, "de_wilcoxon_sig_jaccard": 0.0,
            "de_wilcoxon_lfc_nmae": None,
        }}

    undefined = {"HCT116": ["de_wilcoxon_direction_reach_raw", "de_wilcoxon_lfc_nmae"]}
    source_sha = file_sha256(parent / "split" / "train.h5ad")
    reference_sha = file_sha256(parent / "split" / "reference.h5ad")
    controls_sha = file_sha256(parent / "split" / "input_controls.h5ad")

    def score(path: Path, prediction: Path, values: dict) -> None:
        write_json(path, {
            "implementation": "cell-eval2", "required_version": "0.22.0", "preset": "vcc2026",
            "metrics": list(SCORED_METRICS), "no_target_genes": False,
            "official_leaderboard_score": False, "local_reference_calibration": False,
            "prediction_sha256": file_sha256(prediction), "reference_sha256": reference_sha,
            "raw_metrics": values, "undefined_metrics": undefined,
        })

    parent_plan = parent / "experiment-plan.json"
    write_json(parent_plan, {"training_targets": training, "heldout_targets": heldout})
    baseline_prediction = parent / "control-baseline.h5ad"
    baseline_prediction.write_bytes(b"baseline")
    score(parent / "control-baseline-score" / "scoring-provenance.json", baseline_prediction, metric_values(10))
    no_prior_model = parent / "no-prior" / "model" / "model.pt"
    no_prior_model.parent.mkdir(parents=True)
    no_prior_model.write_bytes(b"model")
    no_prior_prediction = previous / "no-prior" / "diagnostic-strength-1.h5ad"
    no_prior_prediction.parent.mkdir(parents=True)
    no_prior_prediction.write_bytes(b"prediction")
    no_prior = {
        "model_sha256": file_sha256(no_prior_model),
        "prediction_sha256": file_sha256(no_prior_prediction),
        "raw_metrics": metric_values(11), "undefined_metrics": undefined,
    }
    score(previous / "no-prior" / "score" / "scoring-provenance.json", no_prior_prediction, metric_values(11))
    previous_plan = previous / "diagnostic-plan.json"
    write_json(previous_plan, {
        "parent_plan_sha256": file_sha256(parent_plan), "source_sha256": source_sha,
        "reference_sha256": reference_sha, "controls_sha256": controls_sha,
        "diagnostic_strength": 1.0, "calibration_status": "failed: denominator nonpositive",
    })
    previous_result = previous / "result.json"
    write_json(previous_result, {
        "control-baseline": metric_values(10), "no-prior": no_prior, "hct116-epigenome": {},
    })
    shutil.copytree(network_inputs["bundle"], run / "network")
    shuffle_epigenetic_bundle(run / "network", parent / "split" / "train.h5ad",
                              run / "shuffled-network", seed=10)
    plan = {
        "parent_plan_sha256": file_sha256(parent_plan),
        "previous_diagnostic_plan_sha256": file_sha256(previous_plan),
        "previous_results_sha256": file_sha256(previous_result),
        "network_spec_sha256": file_sha256(network_inputs["spec"]),
        "source_sha256": source_sha, "reference_sha256": reference_sha,
        "controls_sha256": controls_sha, "source_archive_sha256": None,
        "arms": ["real-network", "shuffled-network"], "effect_strength": 1.0,
        "shuffle_seed": 10, "prediction_cells_per_target": 2,
    }
    write_json(run / "diagnostic-plan.json", plan)
    result = {"previous_diagnostic": {
        "result_sha256": file_sha256(previous_result), "control_baseline": metric_values(10),
        "no_prior": no_prior, "v1_prior": {},
    }}
    for arm, bundle, mse in (("real-network", "network", 12), ("shuffled-network", "shuffled-network", 13)):
        directory = run / arm
        directory.mkdir(exist_ok=True)
        model = directory / "model" / "model.pt"
        model.parent.mkdir()
        torch.save({
            "training_source_sha256": source_sha, "training_targets": training,
            "epigenetics": {"manifest_sha256": file_sha256(run / bundle / "manifest.json")},
        }, model)
        labels = [target for target in heldout for _ in range(2)]
        prediction = ad.AnnData(
            sp.csr_matrix(np.ones((len(labels), data.n_vars), dtype=np.int32)),
            obs=pd.DataFrame({
                "target_gene": labels,
                "source_control_id": [controls.obs_names[0]] * len(labels),
            }, index=[f"prediction-{i}" for i in range(len(labels))]),
            var=data.var.copy(),
        )
        prediction_path = directory / "diagnostic-strength-1.h5ad"
        prediction.write_h5ad(prediction_path)
        entry = {
            "model_sha256": file_sha256(model), "prediction_sha256": file_sha256(prediction_path),
            "raw_metrics": metric_values(mse), "undefined_metrics": undefined,
        }
        write_json(directory / "result.json", entry)
        score(directory / "score" / "scoring-provenance.json", prediction_path, metric_values(mse))
        result[arm] = entry
    write_json(run / "result.json", result)
    review = review_network_diagnostic(run, parent, previous, network_inputs["spec"])
    assert review["gates"]["artifact_integrity"] == "passed"
    assert review["gates"]["validated_improvement_claim"] == "blocked"
    assert review["scope"]["holdout_previously_scored"] is True
    assert review["reviewer"]["independent_expert_review_performed"] is False
    assert review["methylation_coverage"]["measured_genes"]["promoter_methylation"] == 3
    assert review["comparisons"]["HCT116"]["expr_mse_unbiased_capped_norm"]["improvement_vs_no_response"] == -2
    broken = run / "real-network" / "score" / "scoring-provenance.json"
    metadata = json.loads(broken.read_text(encoding="utf-8"))
    metadata["official_leaderboard_score"] = True
    write_json(broken, metadata)
    with pytest.raises(ValueError, match="Scoring provenance"):
        review_network_diagnostic(run, parent, previous, network_inputs["spec"])


def test_network_links_measured_methylation_accessibility_and_3d_contacts(
    network_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    data = ad.read_h5ad(network_inputs["source"])
    values, masks, metadata = load_epigenetic_bundle(network_inputs["bundle"], data)
    gene = {name: index for index, name in enumerate(data.var_names)}
    feature = {name: index for index, name in enumerate(NETWORK_FEATURES)}
    promoter = feature["promoter_methylation"]
    linked = feature["linked_element_methylation"]
    assert values[gene["G1"], promoter] == pytest.approx(0.5)
    assert values[gene["G2"], promoter] == pytest.approx(0.8)
    assert values[gene["G2"], linked] == pytest.approx(0.5)
    assert masks[gene["G2"], linked] == 1
    assert values[gene["G3"], promoter] == 0 and masks[gene["G3"], promoter] == 1
    assert values[gene["G4"], promoter] == 0 and masks[gene["G4"], promoter] == 0
    assert not masks[gene["AMBIG"]].any()
    edges = pd.read_csv(network_inputs["bundle"] / "graph.csv").set_index(["source", "target"])
    assert edges.loc[("SP1", "G2"), "evidence"] == "contact_tf_atac"
    assert edges.loc[("G1", "G2"), "evidence"] == "contact_promoter"
    assert metadata["format_version"] == 2
    assert "not Hi-C" in metadata["semantics"]["contacts"]
    shuffled = tmp_path / "network-shuffled"
    shuffle_epigenetic_bundle(network_inputs["bundle"], network_inputs["source"], shuffled, seed=10)
    shuffled_values, shuffled_masks, _ = load_epigenetic_bundle(shuffled, data)
    np.testing.assert_array_equal(shuffled_masks, masks)
    for column in range(len(NETWORK_FEATURES)):
        np.testing.assert_allclose(
            np.sort(shuffled_values[masks[:, column] == 1, column]),
            np.sort(values[masks[:, column] == 1, column]),
        )


def test_linked_element_methylation_does_not_require_a_known_tf_peak(
    network_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    spec = json.loads(network_inputs["spec"].read_text(encoding="utf-8"))
    additions = {
        "atac": "chr1\t70000\t70020\n",
        "chromatin_contacts": "chr1\t70000\t70020\tchr1\t60000\t60020\n",
        "methylation": "chr1\t70006\t70007\t0.6\t10\n",
    }
    for source in spec["sources"]:
        if source["role"] in additions:
            path = tmp_path / f"extra-{source['role']}.bed"
            original = Path(source["path"]).read_text(encoding="utf-8")
            path.write_text(original + additions[source["role"]], encoding="utf-8")
            source["path"] = str(path)
            source["sha256"] = file_sha256(path)
    path = tmp_path / "extra-sources.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    bundle = tmp_path / "extra-bundle"
    build_epigenetic_bundle(network_inputs["source"], path, bundle, promoter_flank=50)
    data = ad.read_h5ad(network_inputs["source"])
    values, masks, _ = load_epigenetic_bundle(bundle, data)
    gene = list(data.var_names).index("G4")
    feature = NETWORK_FEATURES.index("linked_element_methylation")
    assert masks[gene, feature] == 1
    assert values[gene, feature] == pytest.approx(0.6)
    edges = pd.read_csv(bundle / "graph.csv")
    assert not ((edges["source"] == "SP1") & (edges["target"] == "G4")).any()


def test_graph_shuffle_handles_high_degree_without_self_edges() -> None:
    genes = np.asarray([f"gene-{i}" for i in range(64)], dtype=object)
    rng = np.random.default_rng(42)
    rows = [
        (source, target, 1, "contact_promoter")
        for source in genes[:32]
        for target in rng.choice(genes[32:], size=24, replace=False)
    ]
    edges = pd.DataFrame(rows, columns=["source", "target", "weight", "evidence"])
    shuffled, swaps = _shuffle_graph_targets(edges, genes, np.random.default_rng(7))
    assert swaps > 0
    assert not (shuffled["source"] == shuffled["target"]).any()
    assert not shuffled.duplicated(["source", "target"]).any()
    assert edges.groupby("source").size().to_dict() == shuffled.groupby("source").size().to_dict()
    assert sorted(edges.groupby("target").size()) == sorted(shuffled.groupby("target").size())
    assert set(zip(edges["source"], edges["target"])) != set(zip(shuffled["source"], shuffled["target"]))


def test_graph_shuffle_rejects_an_unrandomizable_graph() -> None:
    genes = np.asarray(["A", "B"], dtype=object)
    edges = pd.DataFrame([("A", "B", 1, "contact_promoter")],
                         columns=["source", "target", "weight", "evidence"])
    with pytest.raises(ValueError, match="structurally randomized"):
        _shuffle_graph_targets(edges, genes, np.random.default_rng(7))


def test_network_model_propagates_neighbor_methylation_and_reuses_prior(
    network_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    checkpoint_path = _train(network_inputs, tmp_path / "network-model")
    model, checkpoint, graph = load_model(checkpoint_path)
    assert model.config.epi_feature_dim == len(NETWORK_FEATURES)
    assert model.config.epi_graph_propagation and graph is not None
    gene = {name: index for index, name in enumerate(checkpoint["genes"])}
    with torch.no_grad():
        model.epi_adapter[0].weight.zero_()
        model.epi_adapter[0].bias.zero_()
        model.epi_adapter[0].weight[0, 5] = 4
        model.epi_adapter[-1].weight.zero_()
        model.epi_adapter[-1].bias.zero_()
        model.epi_adapter[-1].weight[0, 0] = 5
    target = torch.tensor([gene["SP1"]])
    before = model.target_embedding(target, graph)
    before_without_graph = model.target_embedding(target, None)
    model.epi_values[gene["G2"], 5] = 0.1
    after = model.target_embedding(target, graph)
    assert not torch.allclose(before, after)
    assert torch.allclose(before_without_graph, model.target_embedding(target, None))
    reused = train(
        network_inputs["source"], tmp_path / "network-reuse", latent_dim=4, hidden_dim=6,
        pretrain_steps=2, flow_steps=3, batch_size=4, seed=4, _prior_checkpoint=checkpoint,
    )
    inner, inner_checkpoint, inner_graph = load_model(reused)
    assert inner.config.epi_graph_propagation
    assert torch.equal(inner.epi_values, checkpoint["model_state"]["epi_values"])
    assert torch.equal(inner_graph.coalesce().values(), graph.coalesce().values())
    assert inner_checkpoint["epigenetics"] == checkpoint["epigenetics"]


def test_network_rejects_mismatched_methylation_and_fractional_coverage(
    network_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    spec = json.loads(network_inputs["spec"].read_text(encoding="utf-8"))
    spec["sources"][-1]["context"] = "HFFc6"
    altered = tmp_path / "bad-network.json"
    altered.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="Context or treatment mismatch"):
        build_epigenetic_bundle(network_inputs["source"], altered, tmp_path / "bad-network")
    spec["sources"][-1]["context"] = "HCT116"
    network_inputs["methylation"].write_text("chr1\t30006\t30007\t0.3\t2.5\n", encoding="utf-8")
    spec["sources"][-1]["sha256"] = file_sha256(network_inputs["methylation"])
    altered.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid CpG methylation value"):
        build_epigenetic_bundle(network_inputs["source"], altered, tmp_path / "bad-cpg")


def test_features_masks_and_loop_message_direction(epi_inputs: dict[str, Path]) -> None:
    data = ad.read_h5ad(epi_inputs["source"])
    values, masks, metadata = load_epigenetic_bundle(epi_inputs["bundle"], data)
    gene = {name: index for index, name in enumerate(data.var_names)}
    feature = {name: index for index, name in enumerate(FEATURES)}
    assert values[gene["G1"], feature["atac"]] == 1
    assert values[gene["G1"], feature["sp1"]] == 1
    assert values[gene["G2"], feature["ctcf"]] == 1
    assert values[gene["G1"], feature["hic_loop"]] == 1
    assert values[gene["G2"], feature["hic_loop"]] == 1
    assert not masks[:, feature["usf1"]].any()
    assert not masks[gene["AMBIG"]].any()
    edges = pd.read_csv(epi_inputs["bundle"] / "graph.csv").set_index(["source", "target"])
    assert edges.loc[("SP1", "G1"), "evidence"] == "promoter_tf_atac"
    assert edges.loc[("SP1", "G2"), "evidence"] == "hic_tf_atac"
    assert edges.loc[("G1", "G2"), "evidence"] == "hic_promoter_contact"
    assert edges.loc[("G2", "G1"), "evidence"] == "hic_promoter_contact"
    assert ("G1", "SP1") not in edges.index
    assert metadata["mapping"]["unmapped_or_ambiguous_genes"] == 1
    assert metadata["loop_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("assembly", "GRCh37", "Assembly mismatch"),
        ("context", "HEK293T", "Context or treatment mismatch"),
        ("condition", "drug-treated", "Context or treatment mismatch"),
        ("coordinates", "one-based-inclusive", "Peak track metadata"),
    ],
)
def test_source_mismatch_fails(
    epi_inputs: dict[str, Path], tmp_path: Path, field: str, value: str, message: str,
) -> None:
    spec = json.loads(epi_inputs["spec"].read_text(encoding="utf-8"))
    spec["sources"][1][field] = value
    altered = tmp_path / f"bad-{field}.json"
    altered.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_epigenetic_bundle(epi_inputs["source"], altered, tmp_path / f"bad-{field}")


def test_changed_sources_bundle_and_gene_metadata_fail(
    epi_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    spec = json.loads(epi_inputs["spec"].read_text(encoding="utf-8"))
    Path(spec["sources"][1]["path"]).write_text("chr1\t1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or changed"):
        build_epigenetic_bundle(epi_inputs["source"], epi_inputs["spec"], tmp_path / "changed-source")
    data = ad.read_h5ad(epi_inputs["source"])
    data.var.loc["G1", "ensembl_ids"] = "ENSG99999999998"
    with pytest.raises(ValueError, match="gene axis or Ensembl"):
        load_epigenetic_bundle(epi_inputs["bundle"], data)
    data = ad.read_h5ad(epi_inputs["source"])
    (epi_inputs["bundle"] / "graph.csv").write_text("source,target,weight,evidence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file changed"):
        load_epigenetic_bundle(epi_inputs["bundle"], data)


def test_randomization_preserves_masks_and_edge_degree(
    epi_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    output = tmp_path / "randomized"
    shuffle_epigenetic_bundle(epi_inputs["bundle"], epi_inputs["source"], output, seed=10)
    data = ad.read_h5ad(epi_inputs["source"])
    values, masks, _ = load_epigenetic_bundle(epi_inputs["bundle"], data)
    random_values, random_masks, metadata = load_epigenetic_bundle(output, data)
    np.testing.assert_array_equal(masks, random_masks)
    for column in range(len(FEATURES)):
        assert random_values[:, column].sum() == values[:, column].sum()
    original = pd.read_csv(epi_inputs["bundle"] / "graph.csv")
    shuffled = pd.read_csv(output / "graph.csv")
    assert original.groupby("source").size().to_dict() == shuffled.groupby("source").size().to_dict()
    assert not (shuffled["source"] == shuffled["target"]).any()
    assert metadata["control"]["kind"] == "within-observed-feature-and-graph-target-shuffle"
    assert metadata["control"]["seed"] == 10
    with pytest.raises(ValueError, match="cannot be used as the real reference"):
        shuffle_epigenetic_bundle(output, epi_inputs["source"], tmp_path / "reshuffled", seed=3)


def test_graph_only_randomization_preserves_exact_features(
    epi_inputs: dict[str, Path], tmp_path: Path,
) -> None:
    output = tmp_path / "graph-only"
    shuffle_epigenetic_bundle(
        epi_inputs["bundle"], epi_inputs["source"], output, seed=10, shuffle_features=False,
    )
    original_features = epi_inputs["bundle"] / "features.csv"
    assert file_sha256(output / "features.csv") == file_sha256(original_features)
    data = ad.read_h5ad(epi_inputs["source"])
    values, masks, metadata = load_epigenetic_bundle(output, data)
    original_values, original_masks, _ = load_epigenetic_bundle(epi_inputs["bundle"], data)
    np.testing.assert_array_equal(values, original_values)
    np.testing.assert_array_equal(masks, original_masks)
    original = pd.read_csv(epi_inputs["bundle"] / "graph.csv")
    shuffled = pd.read_csv(output / "graph.csv")
    assert original.groupby("source").size().to_dict() == shuffled.groupby("source").size().to_dict()
    assert set(zip(original.source, original.target)) != set(zip(shuffled.source, shuffled.target))
    assert not (shuffled["source"] == shuffled["target"]).any()
    assert metadata["control"]["kind"] == "graph-target-shuffle-only"
    assert metadata["control"]["features_shuffled"] is False
    with pytest.raises(ValueError, match="shuffle_features must be a boolean"):
        shuffle_epigenetic_bundle(
            epi_inputs["bundle"], epi_inputs["source"], tmp_path / "invalid",
            seed=10, shuffle_features="false",
        )


def test_graph_only_shuffle_cli_is_opt_in() -> None:
    command = [
        "shuffle-epigenetic-bundle", "--bundle", "real", "--gene-data", "train.h5ad",
        "--outdir", "randomized",
    ]
    assert not build_parser().parse_args(command).graph_only
    assert build_parser().parse_args([*command, "--graph-only"]).graph_only


def test_adapter_receives_gradient_without_updating_gene_encoder(epi_inputs: dict[str, Path]) -> None:
    data = ad.read_h5ad(epi_inputs["source"])
    values, masks, _ = load_epigenetic_bundle(epi_inputs["bundle"], data)
    torch.manual_seed(7)
    model = C3VCell(ModelConfig(data.n_vars, latent_dim=4, hidden_dim=6, epi_feature_dim=5))
    model.set_epigenetic_features(torch.from_numpy(values), torch.from_numpy(masks))
    before = model.epi_adapter[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(list(model.velocity.parameters()) + list(model.epi_adapter.parameters()))
    counts = torch.tensor(data.X[:4].toarray(), dtype=torch.float32)
    loss, _ = model.flow_loss(counts, counts.roll(1, 0), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert model.encoder.value.weight.grad is None
    assert model.epi_adapter[-1].weight.grad is not None
    assert torch.count_nonzero(model.epi_adapter[-1].weight.grad) > 0
    optimizer.step()
    assert not torch.equal(before, model.epi_adapter[-1].weight)


def test_adapter_initialization_does_not_change_base_training_rng() -> None:
    torch.manual_seed(13)
    baseline = C3VCell(ModelConfig(5, latent_dim=4, hidden_dim=6))
    expected = torch.rand(8)
    torch.manual_seed(13)
    epi = C3VCell(ModelConfig(5, latent_dim=4, hidden_dim=6, epi_feature_dim=5))
    observed = torch.rand(8)
    assert torch.equal(baseline.encoder.value.weight, epi.encoder.value.weight)
    assert torch.equal(expected, observed)


def test_training_prediction_and_context_guards(epi_inputs: dict[str, Path], tmp_path: Path) -> None:
    checkpoint_path = _train(epi_inputs, tmp_path / "model")
    model, checkpoint, graph = load_model(checkpoint_path)
    assert model.config.epi_feature_dim == 5
    assert checkpoint["epigenetics"]["context"] == "HCT116"
    assert graph is not None
    assert torch.count_nonzero(model.epi_adapter[-1].weight) > 0
    assert torch.allclose(graph.to_dense().abs().sum(dim=1), torch.ones(model.config.n_genes))
    output = tmp_path / "prediction.h5ad"
    predict_holdout(
        checkpoint_path, epi_inputs["reference"], epi_inputs["controls"], output,
        cells_per_target=4, steps=2,
    )
    result = ad.read_h5ad(output)
    assert result.n_obs == 8 and result.n_vars == model.config.n_genes
    for field in ("context", "assembly", "ensembl"):
        altered = ad.read_h5ad(epi_inputs["controls"])
        if field == "context":
            altered.obs["context"] = "HEK293T"
        elif field == "assembly":
            altered.uns["c3vcell_genome_assembly"] = "GRCh37"
        else:
            altered.var.loc["G1", "ensembl_ids"] = "ENSG99999999998"
        path = tmp_path / f"bad-{field}.h5ad"
        altered.write_h5ad(path)
        with pytest.raises(ValueError, match="context|assembly|gene axis"):
            predict_holdout(
                checkpoint_path, epi_inputs["reference"], path,
                tmp_path / f"bad-{field}-prediction.h5ad", cells_per_target=4, steps=2,
            )


def test_nested_calibration_reuses_exact_prior_and_blocks_mismatched_ensemble(
    epi_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = _train(epi_inputs, tmp_path / "model")
    profile = tmp_path / "profile.h5ad"
    fit_mean_response(epi_inputs["source"], profile)

    def score(path: Path, *_args, **_kwargs) -> dict:
        if path.name.startswith("strength-"):
            value = 0.9 if path.name != "strength-00.h5ad" else 1.0
        else:
            value = 1.1
        return {"raw_metrics": {"HCT116": {OBJECTIVE: value}}}

    monkeypatch.setattr("c3vcell.shared_calibration.score_local", score)
    result = calibrate_shared_residual(
        epi_inputs["source"], checkpoint_path, profile, tmp_path / "calibration",
        folds=2, strengths=(0, 1), include_no_response=True,
        cells_per_target=4, prediction_steps=2,
    )
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    plan = json.loads((result.parent / "calibration-plan.json").read_text(encoding="utf-8"))
    assert plan["reuses_checkpoint_graph"]
    assert plan["epigenomic_manifest_sha256"] == final["epigenetics"]["manifest_sha256"]
    for fold in range(2):
        inner = torch.load(
            result.parent / f"fold-{fold:02d}" / "model" / "model.pt",
            map_location="cpu", weights_only=True,
        )
        assert inner["epigenetics"] == final["epigenetics"]
        assert inner["graph"]["shape"] == final["graph"]["shape"]
        assert torch.equal(inner["graph"]["indices"], final["graph"]["indices"])
        assert torch.equal(inner["graph"]["values"], final["graph"]["values"])
        assert torch.equal(inner["model_state"]["epi_values"], final["model_state"]["epi_values"])
        assert torch.equal(inner["model_state"]["epi_masks"], final["model_state"]["epi_masks"])
        assert set(inner["training_targets"]).isdisjoint(plan["fold_targets"][fold])
    output = tmp_path / "shared.h5ad"
    predict_mean_response(
        profile, epi_inputs["reference"], epi_inputs["controls"], output,
        checkpoint_path=checkpoint_path, conditional_calibration_path=result,
        cells_per_target=4, steps=2,
    )
    assert ad.read_h5ad(output).n_obs == 8
    second = _train(epi_inputs, tmp_path / "model-two", seed=4)
    _check_members([checkpoint_path, second])
    altered = torch.load(second, map_location="cpu", weights_only=True)
    altered["model_state"]["epi_values"][0, 0] = 1 - altered["model_state"]["epi_values"][0, 0]
    bad = tmp_path / "different-prior.pt"
    torch.save(altered, bad)
    with pytest.raises(ValueError, match="epigenomic prior"):
        _check_members([checkpoint_path, bad])


def test_effect_calibration_reuses_graph(
    epi_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = _train(epi_inputs, tmp_path / "model")
    monkeypatch.setattr(
        "c3vcell.calibration.score_local",
        lambda *_args, **_kwargs: {"raw_metrics": {"HCT116": {OBJECTIVE: 1.0}}},
    )
    calibrate_effect(
        epi_inputs["source"], checkpoint_path, tmp_path / "effect-calibration",
        folds=2, strengths=(0, 1), cells_per_target=4, prediction_steps=2,
    )
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for fold in range(2):
        inner = torch.load(
            tmp_path / "effect-calibration" / f"fold-{fold:02d}" / "model" / "model.pt",
            map_location="cpu", weights_only=True,
        )
        assert torch.equal(inner["graph"]["values"], final["graph"]["values"])
        assert inner["epigenetics"] == final["epigenetics"]
