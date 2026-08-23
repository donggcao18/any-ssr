import json
import tempfile
import unittest
from pathlib import Path

from expert_selection.config import ExperimentConfig
from expert_selection.ranking import assign_ranks, write_outputs
from expert_selection.records import ScoreRecord
from expert_selection.run import dry_run_report


def record(source, source_index, score, status="ok"):
    return ScoreRecord(
        "run", ["CONCODE", "CodeTrans", "BFP"], "order", "BFP", 2,
        source, source_index, f"anamoe/{source}/0", "gmm", score, None,
        status, 1234, diagnostics={"gmm_mean_log_likelihood": -4.0, "gmm_mean_nll": 4.0},
    )


class RankingTests(unittest.TestCase):
    def test_ranking_descends_with_chronological_tie_break(self):
        records = [record("CodeTrans", 1, 0.5), record("CONCODE", 0, 0.5), record("bad", 2, None, "failed")]
        assign_ranks(records)
        self.assertEqual(records[1].rank, 1)
        self.assertEqual(records[0].rank, 2)
        self.assertIsNone(records[2].rank)

    def test_dry_run_does_not_load_dataset_or_model(self):
        config = ExperimentConfig(targets=("BFP",), methods=("gmm",), adapter_root=Path("definitely_missing"), dry_run=True)
        report = dry_run_report(config, "chronological_sequence")
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["targets"]["BFP"]["dataset_count_status"], "unknown_offline_dry_run")

    def test_output_schema_and_no_row_data(self):
        config = ExperimentConfig(targets=("BFP",), methods=("gmm",), task_order=("CONCODE", "CodeTrans", "BFP"))
        records = [record("CONCODE", 0, 0.9), record("CodeTrans", 1, 0.8)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            write_outputs(output, config, records, [], {"BFP": {"pool_count": 10}}, [])
            first = json.loads((output / "scores_long.jsonl").read_text().splitlines()[0])
            self.assertIn("task_order_id", first)
            self.assertNotIn("row_ids", json.dumps(first))
            header = (output / "scores_wide.csv").read_text().splitlines()[0]
            self.assertIn("gmm_score", header)


if __name__ == "__main__":
    unittest.main()
