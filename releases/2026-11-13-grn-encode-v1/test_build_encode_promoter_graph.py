from __future__ import annotations

from collections import Counter
import gzip
from pathlib import Path
import tempfile
import unittest

import build_encode_promoter_graph as graph


def row(kind, start, end, strand, identifier, symbol, chrom="chr1"):
    return (f'{chrom}\tGENCODE\t{kind}\t{start}\t{end}\t.\t{strand}\t.\t'
            f'gene_id "{identifier}.1"; gene_name "{symbol}";\n')


class GraphTests(unittest.TestCase):
    def test_exact_annotation_mapping_and_stranded_transcript_tss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.gtf.gz"
            with gzip.open(path, "wt", encoding="utf-8") as out:
                out.write(row("gene", 101, 200, "+", "ENSG1", "A"))
                out.write(row("transcript", 110, 150, "+", "ENSG1", "A"))
                out.write(row("gene", 301, 500, "-", "ENSG2", "B"))
                out.write(row("transcript", 330, 490, "-", "ENSG2", "B"))
                out.write(row("gene", 301, 500, "-", "ENSG2", "B", chrom="chrX"))
                out.write(row("gene", 301, 500, "-", "ENSG3", "C", chrom="chr2"))
                out.write(row("transcript", 330, 490, "-", "ENSG3", "C", chrom="chr2"))
            mapped, summary = graph.gene_tss(
                path, {"A": "ENSG1", "B": "ENSG2", "C": "ENSG3"}, {"ENSG1", "ENSG2", "ENSG3"}
            )
            self.assertEqual(mapped["A"]["tss_zero_based"], [109])
            self.assertEqual(mapped["C"]["tss_zero_based"], [489])
            self.assertNotIn("B", mapped)
            self.assertEqual(summary["excluded_features_by_reason"]["ambiguous_or_mismatched_gene"], 1)

    def test_interval_index_handles_nested_and_half_open_boundaries(self):
        index = graph.IntervalIndex([("chr1", 0, 100, "long"), ("chr1", 10, 20, "short"),
                                     ("chr1", 101, 110, "separate")])
        self.assertEqual({value for _, _, value in index.overlaps("chr1", 20, 21)}, {"long"})
        self.assertEqual(list(index.overlaps("chr1", 100, 101)), [])

    def test_requires_triple_overlap_and_drops_self_edges(self):
        mapped = {"A": {"ensembl_id": "ENSG1", "chromosome": "chr1", "tss_zero_based": [100]},
                  "B": {"ensembl_id": "ENSG2", "chromosome": "chr2", "tss_zero_based": [100]}}
        promoter_index, _ = graph.accessible_promoters(mapped, [("chr1", 110, 120), ("chr2", 110, 120)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "peaks.gz"
            with gzip.open(path, "wt", encoding="utf-8") as out:
                out.write("chr1\t100\t109\tp1\t100\t.\t4\t2\t1\t0\n")
                out.write("chr1\t115\t116\tp2\t100\t.\t4\t2\t1\t0\n")
                out.write("chr2\t115\t116\tp3\t100\t.\t4\t2\t1\t0\n")
            source = {"source_symbol": "A", "source_ensembl_id": "ENSG1",
                      "experiment_accession": "ENCSR000AAA", "file_accession": "ENCFF000AAA"}
            support, summary = graph.construct_edges([(source, path)], mapped, promoter_index)
            self.assertEqual(set(support), {("A", "B")})
            self.assertEqual(support[("A", "B")]["peak_gene_overlaps"], 1)
            self.assertEqual(summary["unique_self_edges_excluded"], 1)

    def test_shuffle_preserves_degrees_and_reports_changed_edges(self):
        real = {(f"S{i}", f"T{4 * i + j}") for i in range(5) for j in range(4)}
        shuffled, details = graph.shuffled_edges(real)
        self.assertGreater(details["changed_fraction"], 0)
        self.assertEqual(details["real_edges_absent_from_shuffled"], len(real - shuffled))
        self.assertEqual(details["successful_swaps"], len(real) * graph.SWAPS_PER_EDGE)
        for axis in (0, 1):
            self.assertEqual(Counter(edge[axis] for edge in real), Counter(edge[axis] for edge in shuffled))
        self.assertEqual(shuffled, graph.shuffled_edges(real)[0])
        self.assertNotEqual(shuffled, graph.shuffled_edges(real, seed=20261114)[0])
        with self.assertRaisesRegex(ValueError, "could not complete"):
            graph.shuffled_edges({("A", "B"), ("A", "C")})


if __name__ == "__main__":
    unittest.main()
