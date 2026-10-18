"""CPU-only tests for inference-load telemetry parsing and arithmetic."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch


SPEC = importlib.util.spec_from_file_location(
    "measure_inference_load", Path(__file__).with_name("measure-inference-load.py")
)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)

METRICS = (
    "gpu_util_percent", "gpu_memory_mib", "host_cpu_percent",
    "process_cpu_percent_one_core", "host_ram_used_mib",
    "process_rss_mib", "host_swap_used_mib",
)


class MeasureInferenceLoadTests(unittest.TestCase):
    def setUp(self):
        self.proc = {
            "/proc/stat": "cpu 100 20 30 400 50 6 7 8 900 800\ncpu0 999 999\n",
            "/proc/meminfo": (
                "MemTotal: 8192 kB\nMemFree: 1024 kB\nMemAvailable: 3072 kB\n"
                "SwapTotal: 4096 kB\nSwapFree: 1024 kB\nHugePages_Total: 0\n"
            ),
        }

        def read_text(path, *args, **kwargs):
            value = self.proc[str(path)]
            if isinstance(value, Exception):
                raise value
            return value

        for owner, name, options in (
            (collector.subprocess, "Popen", {
                "side_effect": AssertionError("external processes forbidden"),
            }),
            (collector.subprocess, "check_output", {
                "side_effect": AssertionError("GPU queries forbidden"),
            }),
            (collector.Path, "read_text", {"autospec": True, "side_effect": read_text}),
            (collector.time, "monotonic", {"return_value": 12.5}),
            (collector.os, "sysconf", {
                "side_effect": {"SC_PAGE_SIZE": 4096, "SC_CLK_TCK": 100}.__getitem__,
            }),
        ):
            patcher = patch.object(owner, name, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_snapshot_host_counters_and_memory_without_pid(self):
        self.assertEqual(collector.snapshot(None), {
            "monotonic": 12.5,
            # Guest counters are already included in user/nice; do not add again.
            "cpu_total_ticks": 621,
            "cpu_idle_ticks": 450,
            "host_ram_total_mib": 8,
            "host_ram_used_mib": 5,
            "host_swap_used_mib": 3,
            "process_cpu_ticks": None,
            "process_rss_mib": None,
        })

    def test_snapshot_process_stat_field_numbers_and_comm(self):
        # Use Linux's one-based field numbers, independently of parser offsets.
        fields = {number: str(1000 + number) for number in range(3, 53)}
        fields.update({3: "R", 14: "120", 15: "30", 23: "987654321", 24: "512"})
        suffix = " ".join(fields[number] for number in range(3, 53))
        for comm in ("worker", "worker thread", "worker (nested) end", "a) b(c))"):
            with self.subTest(comm=comm):
                self.proc["/proc/123/stat"] = f"123 ({comm}) {suffix}\n"
                result = collector.snapshot(123)
                self.assertEqual(result["process_cpu_ticks"], 150)
                self.assertEqual(result["process_rss_mib"], 2)

    def test_snapshot_uses_runtime_page_size(self):
        self.proc["/proc/123/stat"] = "123 (worker) R " + "0 " * 20 + "256\n"
        with patch.object(collector.os, "sysconf", return_value=16384) as sysconf:
            self.assertEqual(collector.snapshot(123)["process_rss_mib"], 4)
        sysconf.assert_called_once_with("SC_PAGE_SIZE")

    def test_snapshot_disappearing_process_keeps_host_data(self):
        expected = collector.snapshot(None)
        for error in (FileNotFoundError(), ProcessLookupError()):
            with self.subTest(error=type(error).__name__):
                self.proc["/proc/123/stat"] = error
                self.assertEqual(collector.snapshot(123), expected)

    def test_cpu_rates_one_core_vs_host_normalization(self):
        previous = dict(monotonic=10, cpu_total_ticks=1000,
                        cpu_idle_ticks=500, process_cpu_ticks=50)
        for process_delta, percent in ((0, 0), (200, 100), (400, 200)):
            with self.subTest(process_delta=process_delta):
                # Four CPUs over two seconds: 800 host ticks at 100 Hz.
                current = dict(monotonic=12, cpu_total_ticks=1800,
                               cpu_idle_ticks=1100, process_cpu_ticks=50 + process_delta)
                self.assertEqual(collector.cpu_rates(previous, current), {
                    "host_cpu_percent": 25,
                    "process_cpu_percent_one_core": percent,
                })
        with patch.object(collector.os, "sysconf", return_value=200) as sysconf:
            self.assertEqual(collector.cpu_rates(previous, current)[
                "process_cpu_percent_one_core"], 100)
        sysconf.assert_called_once_with("SC_CLK_TCK")

    def test_cpu_rates_missing_process_counters(self):
        for before, after in ((None, 200), (100, None), (None, None)):
            with self.subTest(before=before, after=after):
                previous = dict(monotonic=10, cpu_total_ticks=1000,
                                cpu_idle_ticks=500, process_cpu_ticks=before)
                current = dict(monotonic=12, cpu_total_ticks=1800,
                               cpu_idle_ticks=1100, process_cpu_ticks=after)
                self.assertEqual(collector.cpu_rates(previous, current), {
                    "host_cpu_percent": 25,
                    "process_cpu_percent_one_core": None,
                })

    def test_cpu_rates_nonpositive_denominators_are_independent(self):
        previous = dict(monotonic=10, cpu_total_ticks=1000,
                        cpu_idle_ticks=500, process_cpu_ticks=100)
        for elapsed in (0, -1, 2):
            for total in (0, -1, 800):
                with self.subTest(elapsed=elapsed, total=total):
                    current = dict(monotonic=10 + elapsed, cpu_total_ticks=1000 + total,
                                   cpu_idle_ticks=1100, process_cpu_ticks=300)
                    self.assertEqual(collector.cpu_rates(previous, current), {
                        "host_cpu_percent": 25 if total > 0 else None,
                        "process_cpu_percent_one_core": 100 if elapsed > 0 else None,
                    })

    def test_aggregate_mixed_missing_values_and_zero(self):
        samples = [
            {"gpu_util_percent": 0, "gpu_memory_mib": 100, "host_cpu_percent": None},
            {"gpu_util_percent": None, "host_cpu_percent": 25, "process_rss_mib": 4},
            {"gpu_util_percent": 60, "gpu_memory_mib": 200, "host_cpu_percent": 75},
            {"elapsed_seconds": 999, "host_swap_used_mib": 0},
        ]
        expected = dict.fromkeys(METRICS)
        expected.update({
            "sample_count": 4,
            "gpu_util_percent": {"min": 0, "mean": 30, "max": 60},
            "gpu_memory_mib": {"min": 100, "mean": 150, "max": 200},
            "host_cpu_percent": {"min": 25, "mean": 50, "max": 75},
            "process_rss_mib": {"min": 4, "mean": 4, "max": 4},
            "host_swap_used_mib": {"min": 0, "mean": 0, "max": 0},
        })
        self.assertEqual(collector.aggregate(samples), expected)

    def test_aggregate_each_metric_filters_missing_without_dropping_zero(self):
        for key in METRICS:
            with self.subTest(metric=key):
                expected = dict.fromkeys(METRICS)
                expected.update(sample_count=4)
                expected[key] = {"min": 0, "mean": 1.5, "max": 3}
                self.assertEqual(collector.aggregate([{key: 0}, {}, {key: None}, {key: 3}]),
                                 expected)

    def test_aggregate_empty_and_all_missing(self):
        for samples in ([], [{}], [dict.fromkeys(METRICS), {}]):
            with self.subTest(samples=samples):
                expected = dict.fromkeys(METRICS)
                expected["sample_count"] = len(samples)
                self.assertEqual(collector.aggregate(samples), expected)


class MeasureInferenceLoadMainTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "never-executed"
        self.binary.write_bytes(b"mock binary")
        self.source = {
            "provenance_sha256": {
                str(self.binary): collector.hashlib.sha256(b"mock binary").hexdigest(),
            },
            "trials": [{
                "name": "probe-262142", "status": "success", "declared_context": 262144,
                "command": [str(self.binary), "--report-file", "old-report.json",
                            "--probe-prompt-tokens", "262142", "--sample-len", "2",
                            "--max-context", "262144", "--ignore-eos"],
            }],
        }
        self.source_path = self.root / "source.json"
        self.source_path.write_text(json.dumps(self.source))
        self.report = {
            "status": "success", "prompt_tokens": 262142, "generated_tokens": 2,
            "forwarded_tokens": 262143, "max_forwarded_position_zero_based": 262142,
            "runtime_context": 262144, "cached_decode_tested": True,
        }
        self.run_count = 0

    def run_main(self, report, *, failure=None, exited=False, succeeds=False, extra_args=()):
        self.run_count += 1
        output = self.root / f"run-{self.run_count}"
        baseline = dict(monotonic=10, cpu_total_ticks=1000, cpu_idle_ticks=500,
                        process_cpu_ticks=None, host_ram_total_mib=8,
                        host_ram_used_mib=5, host_swap_used_mib=0, process_rss_mib=None)
        previous = dict(baseline, monotonic=11, process_cpu_ticks=100, process_rss_mib=2)
        current = dict(previous, monotonic=12, cpu_total_ticks=1400,
                       cpu_idle_ticks=800, process_cpu_ticks=200)
        snapshots = [baseline, previous, current]
        gpu_responses = ["", "", "Mock GPU, 81920, mock\n", "75, 1024\n"]
        process = Mock(pid=123, returncode=0)
        process.poll.side_effect = [None, 0, 0]
        if failure == "snapshot":
            snapshots = [baseline, PermissionError("snapshot denied")]
            process.poll.side_effect = [None]
        elif failure == "query":
            gpu_responses.append(collector.subprocess.TimeoutExpired("nvidia-smi", 10))
            process.poll.side_effect = [None, None, 0 if exited else None]
        if failure and not exited:
            process.returncode = None

        def kill():
            process.returncode = -9

        process.kill.side_effect = kill

        def launch(command, **kwargs):
            self.assertEqual(gpu_query.call_count, 3)
            expected_command = list(self.source['trials'][0]['command'])
            expected_command[2] = str(output / 'inference.json')
            for flag in ('--sample-len', '--max-context'):
                if flag in extra_args:
                    expected_command[expected_command.index(flag) + 1] = extra_args[extra_args.index(flag) + 1]
            self.assertEqual(command, expected_command)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            (output / "inference.json").write_text(json.dumps(report))
            return process

        with patch.object(collector.subprocess, "check_output", side_effect=gpu_responses) as gpu_query, \
                patch.object(collector.subprocess, "Popen", side_effect=launch) as popen, \
                patch.object(collector, "snapshot", side_effect=snapshots) as snapshot, \
                patch.object(collector.os, "sysconf", return_value=100), \
                patch.object(collector.time, "monotonic", return_value=10), \
                patch.object(collector.time, "sleep"), \
                patch("sys.argv", [str(SPEC.origin), str(self.source_path), str(output), *extra_args]), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            if succeeds:
                collector.main()
            else:
                with self.assertRaises(SystemExit) as raised:
                    collector.main()
                self.assertEqual(raised.exception.code, 1)

        popen.assert_called_once()
        process.wait.assert_called_once_with()
        if failure and not exited:
            process.kill.assert_called_once_with()
        else:
            process.kill.assert_not_called()
        self.assertEqual(snapshot.call_args_list,
                         [call(None), call(123)] + ([] if failure == "snapshot" else [call(123)]))
        idle = call(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                     "--format=csv,noheader"], text=True)
        self.assertEqual(gpu_query.call_args_list[:2], [idle, idle])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(json.loads(stdout.getvalue()), summary)
        self.assertEqual(json.loads((output / "source-summary.json").read_text()), self.source)
        self.assertEqual(summary["verified_provenance_sha256"], self.source["provenance_sha256"])
        self.assertEqual(summary["inference"], report)
        self.assertEqual(summary["status"], "success" if succeeds else "failed")
        self.assertEqual(summary["exit_code"], -9 if failure and not exited else 0)
        self.assertFalse(summary["timed_out"])
        self.assertIn("finished_utc", summary)
        samples = [json.loads(line) for line in (output / "telemetry.jsonl").read_text().splitlines()]
        self.assertEqual(len(samples), 0 if failure == "snapshot" else 1)
        for key in ("whole_process", "final_60_seconds"):
            self.assertEqual(summary[key], collector.aggregate(samples))
        return summary, samples

    def test_main_requires_exact_success_accounting(self):
        summary, samples = self.run_main(self.report, succeeds=True)
        self.assertIsNone(summary["telemetry_error"])
        self.assertEqual(samples[0]["gpu_util_percent"], 75)
        self.assertEqual(samples[0]["gpu_memory_mib"], 1024)
        self.assertEqual(samples[0]["host_cpu_percent"], 25)
        self.assertEqual(samples[0]["process_cpu_percent_one_core"], 100)
        for field, wrong in (
            ("status", "failed"), ("prompt_tokens", 262141), ("generated_tokens", 1),
            ("forwarded_tokens", 262142), ("max_forwarded_position_zero_based", 262141),
            ("runtime_context", 262143), ("cached_decode_tested", 1),
        ):
            for missing in (False, True):
                with self.subTest(field=field, missing=missing):
                    report = dict(self.report)
                    if missing:
                        report.pop(field)
                    else:
                        report[field] = wrong
                    summary, _ = self.run_main(report)
                    self.assertIsNone(summary["telemetry_error"])

    def test_main_initial_snapshot_failure_cleans_up_and_persists(self):
        summary, _ = self.run_main(self.report, failure="snapshot")
        self.assertEqual(summary["telemetry_error"], "PermissionError: snapshot denied")
        self.assertEqual(summary["whole_process"], dict.fromkeys(METRICS) | {"sample_count": 0})

    def test_main_variable_output_and_total_budget(self):
        command = self.source['trials'][0]['command']
        command[command.index('--probe-prompt-tokens') + 1] = '32768'
        command[command.index('--max-context') + 1] = '32770'
        self.source_path.write_text(json.dumps(self.source))
        report = dict(self.report, prompt_tokens=32768, generated_tokens=229376)
        self.run_main(report, succeeds=True,
                      extra_args=('--sample-len', '229376', '--max-context', '262144'))
        self.run_main(dict(report, generated_tokens=2),
                      extra_args=('--sample-len', '229376', '--max-context', '262144'))

    def test_main_invalid_budget_rejected_before_gpu_access(self):
        for extra in (('--sample-len', '1'), ('--sample-len', '3'),
                      ('--max-context', '262145')):
            with self.subTest(extra=extra), \
                    patch('sys.argv', [str(SPEC.origin), str(self.source_path),
                                      str(self.root / 'unused'), *extra]), \
                    patch.object(collector.subprocess, 'check_output') as query, \
                    patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as raised:
                    collector.main()
                self.assertEqual(raised.exception.code, 2)
                query.assert_not_called()

    def test_main_telemetry_timeout_preserves_samples_and_forces_failure(self):
        for exited in (False, True):
            with self.subTest(process_already_exited=exited):
                summary, _ = self.run_main(self.report, failure="query", exited=exited)
                self.assertIn("TimeoutExpired:", summary["telemetry_error"])
                self.assertIn("nvidia-smi", summary["telemetry_error"])
                self.assertEqual(summary["whole_process"]["gpu_util_percent"],
                                 {"min": 75, "mean": 75, "max": 75})


if __name__ == "__main__":
    unittest.main()
