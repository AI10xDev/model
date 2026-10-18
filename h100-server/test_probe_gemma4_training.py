"""CPU-only regression tests for the training-capacity probe."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, call


SPEC = importlib.util.spec_from_file_location(
    "probe_gemma4_training", Path(__file__).with_name("probe-gemma4-training.py")
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class SearchTests(unittest.TestCase):
    def test_doubling_then_resolution_aligned_refinement(self):
        trial = Mock(side_effect=lambda length: "success" if length <= 352 else "cuda_oom")
        self.assertEqual(probe.search(trial, 128, 1024, 32), (352, 384, True))
        self.assertEqual(trial.call_args_list, [call(n) for n in (128, 256, 512, 384, 320, 352)])

    def test_cap_is_probed_once_without_claiming_oom(self):
        trial = Mock(return_value="success")
        self.assertEqual(probe.search(trial, 128, 416, 32), (416, None, True))
        self.assertEqual(trial.call_args_list, [call(n) for n in (128, 256, 416)])

    def test_start_at_cap(self):
        trial = Mock(return_value="success")
        self.assertEqual(probe.search(trial, 128, 128, 32), (128, None, True))
        trial.assert_called_once_with(128)

    def test_oom_at_first_size_refines_downward(self):
        trial = Mock(side_effect=lambda length: "success" if length <= 64 else "cuda_oom")
        self.assertEqual(probe.search(trial, 128, 1024, 32), (64, 96, True))
        self.assertEqual(trial.call_args_list, [call(n) for n in (128, 64, 96)])

    def test_oom_at_every_size_has_no_successful_lower_bound(self):
        trial = Mock(return_value="cuda_oom")
        self.assertEqual(probe.search(trial, 128, 1024, 32), (0, 32, True))
        self.assertEqual(trial.call_args_list, [call(n) for n in (128, 64, 32)])

    def test_errors_and_timeouts_stop_without_creating_an_oom_bound(self):
        for status in ("error", "timeout", "invalid_result"):
            for prior_success in (False, True):
                with self.subTest(status=status, prior_success=prior_success):
                    statuses = ["success", status] if prior_success else [status]
                    trial = Mock(side_effect=statuses)
                    expected_lower = 128 if prior_success else 0
                    self.assertEqual(
                        probe.search(trial, 128, 1024, 32),
                        (expected_lower, None, False),
                    )
                    self.assertEqual(trial.call_count, len(statuses))

    def test_refinement_errors_preserve_only_established_bounds(self):
        for status in ("error", "timeout", "invalid_result"):
            with self.subTest(status=status):
                trial = Mock(side_effect=["success", "cuda_oom", status])
                self.assertEqual(probe.search(trial, 128, 1024, 32), (128, 256, False))
                self.assertEqual(trial.call_args_list, [call(128), call(256), call(192)])

    def test_trial_exceptions_propagate_instead_of_becoming_oom(self):
        for error in (RuntimeError("failed"), subprocess.TimeoutExpired("not-run", 1)):
            with self.subTest(error=type(error).__name__):
                trial = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    probe.search(trial, 128, 1024, 32)
                trial.assert_called_once_with(128)


class ValidateRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.length, self.steps = 128, 20
        self.checkpoint = self.run / "checkpoint-20"
        self.checkpoint.mkdir()
        self.adapter = self.checkpoint / "adapter.safetensors"
        # Validation hashes adapter bytes; it does not parse safetensors.
        self.adapter.write_bytes(b"updated adapter")
        self.initial_hash = hashlib.sha256(b"initial adapter").hexdigest()
        self.state = {
            "mode": "quantized-lora", "quantization": "Q4_0", "dtype": "bf16",
            "master_dtype": "f32", "optimizer_dtype": "f32", "loss_dtype": "f32",
            "optimizer": "AdamW", "lora_rank": 8, "lora_alpha": 8,
            "batch_size": 1, "gradient_accumulation_steps": 4,
            "learning_rate": 0.0001, "weight_decay": 0, "max_grad_norm": 1,
            "sequence_length": 128, "step": 20, "steps_requested": 20,
            "tokens_seen": 10240, "quantized_projections": 410, "eval_batches": 2,
        }
        self.metrics = [{"step": 0, "validation_loss": 2.0}]
        for step in range(1, 21):
            row = {"step": step, "train_loss": 1.5, "gradient_norm": 0.5,
                   "tokens_seen": step * 512}
            if step % 10 == 0:
                row["validation_loss"] = 1.75
            self.metrics.append(row)
        self.write_evidence()

    def write_evidence(self, state=None, metrics=None):
        (self.checkpoint / "training-state.json").write_text(
            json.dumps(self.state if state is None else state)
        )
        (self.run / "metrics.jsonl").write_text("".join(
            json.dumps(row) + "\n" for row in (self.metrics if metrics is None else metrics)
        ))

    def validate(self):
        return probe.validate_run(self.run, self.length, self.steps, self.initial_hash)

    def test_complete_run_returns_verified_evidence(self):
        result = self.validate()
        self.assertEqual(result["final_metrics"], self.metrics[-1])
        self.assertEqual(result["policy_verified"], self.state)
        self.assertEqual(result["adapter_sha256"], hashlib.sha256(b"updated adapter").hexdigest())

    def test_every_required_state_field_must_match_and_be_present(self):
        for key, value in self.state.items():
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    state = self.state.copy()
                    if missing:
                        del state[key]
                    else:
                        state[key] = "wrong" if isinstance(value, str) else value + 1
                    self.write_evidence(state=state)
                    with self.assertRaisesRegex(ValueError, "policy/completion mismatch"):
                        self.validate()

    def test_incomplete_duplicate_or_out_of_order_updates_are_rejected(self):
        variants = {
            "missing_initial": self.metrics[1:],
            "missing_final": self.metrics[:-1],
            "missing_middle": self.metrics[:5] + self.metrics[6:],
            "duplicate": self.metrics + [self.metrics[-1]],
            "out_of_order": list(reversed(self.metrics)),
            "empty": [],
        }
        for name, metrics in variants.items():
            with self.subTest(name=name):
                self.write_evidence(metrics=metrics)
                with self.assertRaisesRegex(ValueError, "missing optimizer updates"):
                    self.validate()

    def test_required_metrics_must_be_present_numeric_and_finite(self):
        fields = [(0, "validation_loss"), (1, "train_loss"), (1, "gradient_norm"),
                  (10, "validation_loss"), (20, "validation_loss")]
        for step, key in fields:
            for value in (None, "1.0", float("nan"), float("inf"), float("-inf")):
                with self.subTest(step=step, key=key, value=value):
                    metrics = [row.copy() for row in self.metrics]
                    if value is None:
                        del metrics[step][key]
                    else:
                        metrics[step][key] = value
                    self.write_evidence(metrics=metrics)
                    with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
                        self.validate()

    def test_invalid_token_accounting_and_nonpositive_gradients_are_rejected(self):
        for key, value in (("tokens_seen", 511), ("gradient_norm", 0), ("gradient_norm", -1)):
            with self.subTest(key=key, value=value):
                metrics = [row.copy() for row in self.metrics]
                metrics[1][key] = value
                self.write_evidence(metrics=metrics)
                with self.assertRaisesRegex(ValueError, "invalid token accounting or zero gradient"):
                    self.validate()

    def test_missing_checkpoint_or_adapter_is_rejected(self):
        for path in (self.checkpoint / "training-state.json", self.adapter):
            with self.subTest(file=path.name):
                contents = path.read_bytes()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    self.validate()
                path.write_bytes(contents)

    def test_unchanged_adapter_is_rejected(self):
        self.adapter.write_bytes(b"initial adapter")
        with self.assertRaisesRegex(ValueError, "adapter did not change"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
