"""CPU-only sweep tests: mock the monitor, never launch a GPU process."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'sweep_inference_output', Path(__file__).with_name('sweep-inference-output.py'))
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


class SweepTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_path = self.root / 'source.json'
        self.boundary_path = self.root / 'boundary.json'
        self.output = self.root / 'sweep'
        self.provenance = {'/model/config.json': 'config-hash', '/model/binary': 'binary-hash'}
        self.source = {
            'provenance_sha256': dict(self.provenance, **{'/model/probe.py': 'old-script'}),
            'trials': [{
                'name': f'probe-{p}', 'status': 'success', 'declared_context': 262144,
                'prompt_tokens': p, 'generated_tokens': 2, 'forwarded_tokens': p + 1,
                'max_forwarded_position_zero_based': p, 'runtime_context': p + 2,
                'command': ['/model/binary', '--probe-prompt-tokens', str(p),
                            '--sample-len', '2', '--max-context', str(p + 2), '--ignore-eos'],
            } for p in sweep.PROMPTS],
        }
        self.boundary = self.report(262142, 2, 262144)
        self.write_inputs()
        self.calls = []
        self.failures = {}
        self.snapshots = []
        real_save = sweep.save_summary

        def save(path, summary):
            real_save(path, summary)
            self.snapshots.append(json.loads(path.read_text()))

        for owner, name, options in (
            (sweep.subprocess, 'run', {'side_effect': self.monitor}),
            (sweep.subprocess, 'Popen', {'side_effect': AssertionError('no processes')}),
            (sweep.subprocess, 'check_output', {'side_effect': AssertionError('no GPU queries')}),
            (sweep, 'save_summary', {'side_effect': save}),
        ):
            patcher = patch.object(owner, name, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def report(self, prompt, generated, cap):
        return {
            'status': 'success', 'exit_code': 0, 'timed_out': False, 'telemetry_error': None,
            'verified_provenance_sha256': dict(self.provenance),
            'inference': {
                'status': 'success', 'prompt_tokens': prompt, 'generated_tokens': generated,
                'forwarded_tokens': prompt + generated - 1,
                'max_forwarded_position_zero_based': prompt + generated - 2,
                'runtime_context': cap, 'cached_decode_tested': True,
            },
        }

    def write_inputs(self):
        self.source_path.write_text(json.dumps(self.source))
        self.boundary_path.write_text(json.dumps(self.boundary))

    def monitor(self, command, **kwargs):
        self.calls.append(command)
        state = json.loads((self.output / 'summary.json').read_text())
        output = Path(command[3])
        self.assertEqual(state['active_trial'], output.name)
        self.assertEqual(state['status'], 'running')
        self.assertEqual(len(state['completed_results']), len(self.calls) - 1)
        self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
        self.assertFalse(kwargs['check'])
        self.assertEqual(Path(kwargs['stdout'].name), output.with_suffix('.log'))
        failure = self.failures.get(output.name)
        if isinstance(failure, Exception):
            raise failure
        output.mkdir()
        prompt = int(command[command.index('--trial') + 1].split('-')[1])
        generated = int(command[command.index('--sample-len') + 1]) if '--sample-len' in command else 2
        cap = int(command[command.index('--max-context') + 1]) if '--max-context' in command else prompt + 2
        report = self.report(prompt, generated, cap)
        if callable(failure):
            failure(report)
        if failure != 'missing':
            (output / 'summary.json').write_text(
                '{' if failure == 'malformed' else json.dumps(report))
        return subprocess.CompletedProcess(command, 1 if failure == 'exit' else 0)

    def run_main(self, extra=()):
        with patch('sys.stdout', new_callable=io.StringIO):
            return sweep.main([str(self.source_path), str(self.output),
                               '--boundary-summary', str(self.boundary_path), *extra])

    def summary(self):
        return json.loads((self.output / 'summary.json').read_text())

    def test_success_order_commands_exact_budgets_and_durable_states(self):
        self.assertEqual(self.run_main(), 0)
        summary = self.summary()
        self.assertEqual(len(self.calls), 6)
        self.assertEqual([Path(c[3]).name for c in self.calls],
                         [f'{phase}-{p}' for phase in ('input', 'output') for p in sweep.PROMPTS])
        for index, command in enumerate(self.calls):
            p = sweep.PROMPTS[index % 3]
            expected = [sweep.sys.executable, str(Path(SPEC.origin).with_name('measure-inference-load.py')),
                        str(self.output / 'source-summary.json'), command[3],
                        '--trial', f'probe-{p}', '--timeout', '12000' if index < 3 else '172800']
            if index >= 3:
                expected += ['--sample-len', str(262144 - p), '--max-context', '262144']
            self.assertEqual(command, expected)
        self.assertEqual(self.snapshots[0]['status'], 'pending')
        self.assertTrue(all(t['status'] == 'pending' for t in self.snapshots[0]['trials']))
        self.assertEqual(len(self.snapshots), 14)
        self.assertEqual(summary['status'], 'success')
        self.assertIsNone(summary['active_trial'])
        self.assertEqual(len(summary['completed_results']), 6)
        for trial in summary['trials']:
            self.assertEqual(trial['status'], 'success')
            self.assertIn('started_utc', trial)
            self.assertIn('finished_utc', trial)
            self.assertTrue(Path(trial['log_path']).exists())
        for entry in summary['inputs'][:4]:
            self.assertEqual(entry['maximum_status'], 'success')
            self.assertEqual(entry['maximum_output_tokens'], 262144 - entry['prompt_tokens'])
            self.assertTrue(entry['measured_prefill'])
        boundary, exact = summary['inputs'][-2:]
        self.assertTrue(boundary['reused'])
        self.assertEqual(exact['prompt_tokens'], 262144)
        self.assertEqual(exact['architectural_output_budget'], 0)
        self.assertEqual(exact['maximum_status'], 'architectural-budget-only')
        self.assertIsNone(exact['maximum_output_tokens'])
        self.assertFalse(exact['measured_prefill'])
        self.assertEqual((self.output / 'source-summary.json').read_bytes(), self.source_path.read_bytes())
        self.assertEqual((self.output / 'boundary-summary.json').read_bytes(), self.boundary_path.read_bytes())
        self.assertFalse((self.output / 'summary.json.tmp').exists())

    def test_failure_continues_and_never_claims_maximum(self):
        self.failures = {
            'input-32768': OSError('launch denied'),
            'input-65536': 'exit',  # Includes GPU-busy refusal.
            'output-32768': lambda r: r.update(timed_out=True),
            'output-65536': lambda r: r.update(status='failed', exit_code=1, error='CUDA OOM'),
            'output-131072': 'missing',
        }
        self.assertEqual(self.run_main(), 1)
        summary = self.summary()
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(summary['status'], 'inconclusive')
        self.assertIsNone(summary['active_trial'])
        self.assertIn('finished_utc', summary)
        self.assertIn('OSError: launch denied', summary['trials'][0]['error'])
        self.assertEqual(summary['trials'][2]['status'], 'success')
        for entry in summary['inputs'][:3]:
            self.assertEqual(entry['maximum_status'], 'inconclusive')
            self.assertEqual(entry['verified_output_lower_bound'], 2)
            self.assertIsNone(entry['maximum_output_tokens'])

    def test_bad_monitor_accounting_provenance_and_json_are_inconclusive(self):
        self.failures = {
            'output-32768': lambda r: r['inference'].update(generated_tokens=2),
            'output-65536': lambda r: r.update(verified_provenance_sha256={}),
            'output-131072': 'malformed',
        }
        self.assertEqual(self.run_main(), 1)
        self.assertTrue(all(e['maximum_status'] == 'inconclusive' for e in self.summary()['inputs'][:3]))

    def test_custom_timeouts(self):
        self.assertEqual(self.run_main(['--input-timeout', '19', '--output-timeout', '41']), 0)
        self.assertEqual([c[c.index('--timeout') + 1] for c in self.calls], ['19'] * 3 + ['41'] * 3)

    def test_boundary_requires_clean_exact_success_and_matching_provenance(self):
        mutations = [(key, value) for key, value in (
            ('status', 'failed'), ('exit_code', 1), ('timed_out', True),
            ('telemetry_error', 'broken'), ('verified_provenance_sha256', {}),
        )]
        for key, value in mutations:
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    report = copy.deepcopy(self.boundary)
                    if missing:
                        report.pop(key)
                    else:
                        report[key] = value
                    with self.assertRaises(ValueError):
                        sweep.validate_measurement(report, 262142, 2, 262144, self.provenance)
        for key in self.boundary['inference']:
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    report = copy.deepcopy(self.boundary)
                    if missing:
                        report['inference'].pop(key)
                    else:
                        report['inference'][key] = 1
                    with self.assertRaises(ValueError):
                        sweep.validate_measurement(report, 262142, 2, 262144, self.provenance)

    def test_rejects_bad_inputs_before_launch_or_output_creation(self):
        source = copy.deepcopy(self.source)
        boundary = copy.deepcopy(self.boundary)
        for failure in ('context', 'missing-context', 'missing-trial', 'source-count',
                        'command', 'boundary', 'provenance', 'empty-provenance', 'timeout'):
            with self.subTest(failure=failure):
                self.source = copy.deepcopy(source)
                self.boundary = copy.deepcopy(boundary)
                if failure == 'context':
                    self.source['trials'][1]['declared_context'] = 131072
                elif failure == 'missing-context':
                    self.source['trials'][0].pop('declared_context')
                elif failure == 'missing-trial':
                    self.source['trials'].pop()
                elif failure == 'source-count':
                    self.source['trials'][0]['generated_tokens'] = 1
                elif failure == 'command':
                    self.source['trials'][0]['command'][4] = '3'
                elif failure == 'boundary':
                    self.boundary['timed_out'] = True
                elif failure == 'provenance':
                    self.boundary['verified_provenance_sha256']['/model/binary'] = 'wrong'
                elif failure == 'empty-provenance':
                    self.source['provenance_sha256'] = {'probe.py': 'hash'}
                self.write_inputs()
                with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
                    self.run_main(['--input-timeout', '0'] if failure == 'timeout' else [])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.calls, [])

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / 'summary.json'
        sentinel.write_text('existing evidence')
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
            self.run_main()
        self.assertEqual(sentinel.read_text(), 'existing evidence')
        self.assertEqual(self.calls, [])

    def test_atomic_replace_keeps_previous_summary_until_replace(self):
        path = self.root / 'atomic.json'
        path.write_text('{"status": "pending"}')
        replace = sweep.os.replace

        def inspect(temporary, destination):
            self.assertEqual(json.loads(path.read_text())['status'], 'pending')
            self.assertEqual(json.loads(temporary.read_text())['status'], 'running')
            replace(temporary, destination)

        with patch.object(sweep.os, 'replace', side_effect=inspect) as mocked:
            sweep.save_summary(path, {'status': 'running'})
        mocked.assert_called_once()
        self.assertEqual(json.loads(path.read_text())['status'], 'running')


if __name__ == '__main__':
    unittest.main()
