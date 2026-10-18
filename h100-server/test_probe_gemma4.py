"""CPU-only regression tests for the inference-capacity probe."""
import importlib.util
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "probe_gemma4", Path(__file__).with_name("probe-gemma4.py")
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = self.root / "adapter"
        self.source = self.root / "source"
        self.adapter.mkdir()
        self.source.mkdir()
        (self.adapter / "training-state.json").write_text(json.dumps({
            "source_checkpoint": str(self.source),
        }))
        (self.source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": "model-00001.safetensors"},
        }))
        (self.root / "checkpoint-manifest.json").write_text(json.dumps({
            "model_id": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "sha256": {"model-00001.safetensors": "mock-sha256"},
        }))
        self.runs = itertools.count()
        # Patch all external activity, including validation's initial GPU query.
        for owner, name, options in (
            (probe.subprocess, "Popen", {"side_effect": AssertionError("unexpected process")}),
            (probe.subprocess, "check_output", {"side_effect": self.gpu_query}),
            (probe, "sha256", {"return_value": "mock-sha256"}),
            (probe.time, "sleep", {}),
            (probe.time, "monotonic", {"side_effect": itertools.count(step=2)}),
        ):
            patcher = patch.object(owner, name, **options)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if name == "Popen":
                self.popen = mocked
            elif name == "check_output":
                self.check_output = mocked

    def gpu_query(self, command, **kwargs):
        self.assertEqual(command[:3], ["nvidia-smi", "-i", "0"])
        return {
            "--query-compute-apps=pid": "",
            "--query-gpu=name,memory.total,driver_version": "Mock GPU, 81920, mock\n",
            "--query-gpu=memory.used": "1234\n",
        }[command[3]]

    def invoke(self, output, *args):
        argv = [str(SPEC.origin), str(output), "--adapter", str(self.adapter),
                "--binary", str(self.root / "never-executed"), "--timeout", "1", *args]
        with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            try:
                probe.main()
            finally:
                self.stderr = stderr.getvalue()

    def run_probe(self, outcomes, *args, mutate_report=None):
        output = self.root / f"evidence-{next(self.runs)}"
        pending = iter(outcomes)
        processes = []

        def launch(command, stdout, **kwargs):
            report_path = Path(command[command.index("--report-file") + 1])
            synthetic = "--probe-prompt-tokens" in command
            length = int(command[command.index("--probe-prompt-tokens") + 1]) if synthetic else 8
            status = next(pending) if synthetic else "success"
            process = Mock(returncode=0 if status == "success" else 1)
            process.poll.return_value = process.returncode
            if status == "success":
                report = {
                    "status": "success", "prompt_tokens": length,
                    "generated_tokens": 2, "forwarded_tokens": length + 1,
                    "max_forwarded_position_zero_based": length,
                    "runtime_context": length + 2, "cached_decode_tested": True,
                    "declared_context": 262144, "timings_seconds": {},
                    "output_text": "mock generation",
                }
                if synthetic and mutate_report:
                    mutate_report(report)
                report_path.write_text(json.dumps(report))
            elif status == "cuda_oom":
                stdout.write("CUDA_ERROR_OUT_OF_MEMORY\n")
            elif status in ("timeout", "timeout_with_oom"):
                process.poll.return_value = None

                def kill():
                    process.returncode = -9
                    process.poll.return_value = -9

                process.kill.side_effect = kill
                if status == "timeout_with_oom":
                    stdout.write("CUDA_ERROR_OUT_OF_MEMORY\n")
            else:
                self.assertEqual(status, "error")
                stdout.write("non-OOM runtime failure\n")
            processes.append((process, status))
            return process

        self.popen.side_effect = launch
        self.invoke(output, *args)
        self.assertEqual(list(pending), [], "not all expected trials ran")
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(summary["trials"]), len(processes))
        self.assertEqual([t["name"] for t in summary["trials"][:2]],
                         ["generation-training-prefix", "generation-instruction"])
        for trial, (process, status) in zip(summary["trials"], processes):
            process.wait.assert_called_once_with()
            if status.startswith("timeout"):
                process.kill.assert_called_once_with()
                self.assertEqual(trial["sampled_max_gpu_mib"], 1234)
            else:
                process.kill.assert_not_called()
            self.assertTrue((output / (trial["name"] + "-gpu.json")).exists())
        return summary

    def assert_search(self, summary, lengths, lower, upper):
        trials = summary["trials"][2:]
        self.assertEqual([t["name"] for t in trials], [f"probe-{n}" for n in lengths])
        self.assertEqual(summary["largest_successful_prompt_tokens"], lower)
        self.assertEqual(summary["smallest_oom_prompt_tokens"], upper)
        width = upper - lower if upper is not None and lower else None
        self.assertEqual(summary["observed_bracket_width_tokens"], width)

    def test_invalid_start_prompt_rejected_before_gpu_or_output_activity(self):
        for value in ("0", "-1", "1025", "not-an-integer"):
            with self.subTest(start=value):
                output = self.root / "invalid"
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(output, "--max-prompt", "1024", "--start-prompt", value)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("start-prompt", self.stderr)
                self.check_output.assert_not_called()
                self.popen.assert_not_called()
                self.assertFalse(output.exists())

    def test_start_prompt_accepts_both_endpoints(self):
        for start in (1, 1024):
            with self.subTest(start=start):
                summary = self.run_probe(["error"], "--max-prompt", "1024",
                                         "--start-prompt", str(start))
                self.assert_search(summary, [start], None, None)

    def test_default_start_and_doubling_clamp_to_cap(self):
        summary = self.run_probe(["success"] * 3, "--max-prompt", "3000")
        self.assert_search(summary, [1024, 2048, 3000], 3000, None)
        self.assertNotIn("boundary_confirmation", summary)

    def test_start_at_cap_runs_once(self):
        summary = self.run_probe(["success"], "--start-prompt", "2048",
                                 "--max-prompt", "2048")
        self.assert_search(summary, [2048], 2048, None)

    def test_doubling_then_resolution_aligned_refinement(self):
        summary = self.run_probe(
            ["success", "success", "cuda_oom", "cuda_oom", "success", "success"],
            "--start-prompt", "128", "--max-prompt", "1024", "--resolution", "32",
        )
        lengths = [128, 256, 512, 384, 320, 352]
        self.assert_search(summary, lengths, 352, 384)
        self.assertEqual(summary["resolution_tokens"], 32)
        for trial, length in zip(summary["trials"][2:], lengths):
            command = trial["command"]
            for flag, value in (("--probe-prompt-tokens", length),
                                ("--max-context", length + 2), ("--sample-len", 2)):
                self.assertEqual(command[command.index(flag) + 1], str(value))
            self.assertIn("--raw-prompt", command)
            self.assertIn("--ignore-eos", command)

    def test_initial_oom_refines_downward(self):
        summary = self.run_probe(["cuda_oom", "success", "cuda_oom"],
                                 "--start-prompt", "128", "--resolution", "32")
        self.assert_search(summary, [128, 64, 96], 64, 96)

    def test_all_oom_has_no_successful_lower_bound(self):
        summary = self.run_probe(["cuda_oom"] * 3,
                                 "--start-prompt", "128", "--resolution", "32")
        self.assert_search(summary, [128, 64, 32], None, 32)

    def test_errors_and_timeouts_never_establish_oom_bound(self):
        for status in ("error", "timeout", "timeout_with_oom"):
            for prior_success in (False, True):
                with self.subTest(status=status, prior_success=prior_success):
                    outcomes = ["success", status] if prior_success else [status]
                    summary = self.run_probe(outcomes)
                    self.assert_search(summary, [1024, 2048] if prior_success else [1024],
                                       1024 if prior_success else None, None)
                    self.assertEqual(summary["trials"][-1]["status"],
                                     "timeout" if status.startswith("timeout") else "error")

    def test_refinement_errors_preserve_only_established_bounds(self):
        for status in ("error", "timeout", "timeout_with_oom"):
            with self.subTest(status=status):
                summary = self.run_probe(["success", "cuda_oom", status],
                                         "--resolution", "256")
                self.assert_search(summary, [1024, 2048, 1536], 1024, 2048)

    def test_confirm_boundary_repeats_both_in_fresh_processes(self):
        summary = self.run_probe(["success", "cuda_oom", "success", "cuda_oom"],
                                 "--confirm-boundary")
        self.assertEqual([t["name"] for t in summary["trials"][2:]],
                         ["probe-1024", "probe-2048", "confirm-success-1024", "confirm-oom-2048"])
        self.assertEqual(summary["boundary_confirmation"], {
            "success_boundary_status": "success", "oom_boundary_status": "cuda_oom",
            "confirmed_interval": True,
        })
        self.assertEqual(self.popen.call_count, 6)
        paths = [c.args[0][c.args[0].index("--report-file") + 1]
                 for c in self.popen.call_args_list]
        self.assertEqual(len(set(paths)), 6)

    def test_disagreeing_confirmations_do_not_confirm_or_rewrite_bounds(self):
        for success, oom in (("cuda_oom", "cuda_oom"), ("success", "success"),
                             ("error", "cuda_oom"), ("success", "error"),
                             ("timeout", "cuda_oom"), ("success", "timeout_with_oom")):
            with self.subTest(success=success, oom=oom):
                summary = self.run_probe(["success", "cuda_oom", success, oom],
                                         "--confirm-boundary")
                self.assertFalse(summary["boundary_confirmation"]["confirmed_interval"])
                self.assertEqual(summary["largest_successful_prompt_tokens"], 1024)
                self.assertEqual(summary["smallest_oom_prompt_tokens"], 2048)

    def test_confirm_only_existing_boundaries(self):
        for status, prefix, key in (
            ("success", "confirm-success-1024", "success_boundary_status"),
            ("cuda_oom", "confirm-oom-1024", "oom_boundary_status"),
            ("error", None, None),
        ):
            with self.subTest(status=status):
                summary = self.run_probe([status] * (2 if prefix else 1),
                                         "--max-prompt", "1024", "--confirm-boundary")
                expected = {"confirmed_interval": False}
                if key:
                    expected[key] = status
                self.assertEqual(summary["boundary_confirmation"], expected)
                self.assertEqual([t["name"] for t in summary["trials"][2:]],
                                 ["probe-1024"] + ([prefix] if prefix else []))

    def test_malformed_success_accounting_is_saved_as_error_not_capacity(self):
        invalid = {"status": "error", "prompt_tokens": 1023, "generated_tokens": 1,
                   "forwarded_tokens": 1024, "max_forwarded_position_zero_based": 1023,
                   "runtime_context": 1025, "cached_decode_tested": 1}
        for field, wrong in invalid.items():
            for value in (wrong, None):
                with self.subTest(field=field, value=value):
                    summary = self.run_probe(
                        ["success"], mutate_report=lambda r: r.update({field: value}),
                    )
                    self.assert_search(summary, [1024], None, None)
                    trial = summary["trials"][-1]
                    self.assertEqual(trial["exit_code"], 0)
                    self.assertEqual(trial["status"], "error")
                    self.assertIn("exact token/context/decode validation", trial["error_excerpt"])

    def test_missing_success_accounting_is_saved_as_error_not_capacity(self):
        for field in ("status", "prompt_tokens", "generated_tokens", "forwarded_tokens",
                      "max_forwarded_position_zero_based", "runtime_context", "cached_decode_tested"):
            with self.subTest(field=field):
                summary = self.run_probe(["success"], mutate_report=lambda r: r.pop(field))
                self.assert_search(summary, [1024], None, None)
                self.assertEqual(summary["trials"][-1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
