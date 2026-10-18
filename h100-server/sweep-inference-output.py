#!/usr/bin/env python3
"""Serial inference/load sweep; launch this runner with nohup for detached use."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROMPTS = (32768, 65536, 131072)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def save_summary(path, summary):
    summary['updated_utc'] = utc()
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_measurement(summary, prompt, generated, cap, provenance):
    if not isinstance(summary, dict) or not all((
        summary.get('status') == 'success',
        type(summary.get('exit_code')) is int and summary['exit_code'] == 0,
        summary.get('timed_out') is False,
        'telemetry_error' in summary and summary['telemetry_error'] is None,
        summary.get('verified_provenance_sha256') == provenance,
    )):
        raise ValueError('monitor did not establish clean success with matching provenance')
    result = summary.get('inference')
    if not isinstance(result, dict):
        raise ValueError('missing monitor inference accounting')
    expected = {
        'prompt_tokens': prompt, 'generated_tokens': generated,
        'forwarded_tokens': prompt + generated - 1,
        'max_forwarded_position_zero_based': prompt + generated - 2,
        'runtime_context': cap,
    }
    if result.get('status') != 'success' or result.get('cached_decode_tested') is not True:
        raise ValueError('inference did not establish successful cached decode')
    for key, value in expected.items():
        if type(result.get(key)) is not int or result[key] != value:
            raise ValueError(f'inexact inference accounting: {key}, expected {value}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_summary', type=Path)
    parser.add_argument('output', type=Path, help='new directory; never overwrites evidence')
    parser.add_argument('--boundary-summary', required=True, type=Path)
    parser.add_argument('--input-timeout', type=int, default=12000)
    parser.add_argument('--output-timeout', type=int, default=172800)
    args = parser.parse_args(argv)
    if args.input_timeout <= 0 or args.output_timeout <= 0:
        parser.error('timeouts must be positive')
    args.source_summary = args.source_summary.resolve()
    args.boundary_summary = args.boundary_summary.resolve()
    args.output = args.output.resolve()
    try:
        source_bytes = args.source_summary.read_bytes()
        source = json.loads(source_bytes)
        trials = source['trials']
        if not trials or any(type(t.get('declared_context')) is not int
                             or t['declared_context'] != 262144 for t in trials):
            raise ValueError('source trials must consistently declare context 262144')
        cap = trials[0]['declared_context']
        provenance = {k: v for k, v in source['provenance_sha256'].items()
                      if not k.endswith('.py')}
        if not provenance:
            raise ValueError('source has no non-Python provenance')
        for prompt in PROMPTS:
            matches = [t for t in trials if t['name'] == f'probe-{prompt}']
            if len(matches) != 1:
                raise ValueError(f'expected one source trial probe-{prompt}')
            trial = matches[0]
            if trial['status'] != 'success' or any(
                trial.get(key) != value for key, value in (
                    ('prompt_tokens', prompt), ('generated_tokens', 2),
                    ('forwarded_tokens', prompt + 1),
                    ('max_forwarded_position_zero_based', prompt),
                    ('runtime_context', prompt + 2),
                )
            ):
                raise ValueError(f'source probe-{prompt} must establish two outputs')
            command = trial['command']
            for flag, value in (('--probe-prompt-tokens', prompt),
                                ('--sample-len', 2), ('--max-context', prompt + 2)):
                if command.count(flag) != 1 or command[command.index(flag) + 1] != str(value):
                    raise ValueError(f'source probe-{prompt} has unexpected {flag}')
        boundary_bytes = args.boundary_summary.read_bytes()
        boundary = json.loads(boundary_bytes)
        validate_measurement(boundary, cap - 2, 2, cap, provenance)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        parser.error(f'invalid source/boundary evidence: {error}')
    try:
        args.output.mkdir(exist_ok=False)
    except OSError as error:
        parser.error(f'cannot create new output directory: {error}')
    # Retain the exact inputs used for validation, even if their originals change.
    for name, content in (('source-summary.json', source_bytes),
                          ('boundary-summary.json', boundary_bytes)):
        with (args.output / name).open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    source_path = args.output / 'source-summary.json'
    inputs = [{
        'prompt_tokens': prompt, 'architectural_output_budget': cap - prompt,
        'measured_prefill': False, 'maximum_status': 'pending',
        'maximum_output_tokens': None, 'verified_output_lower_bound': 2,
        'lower_bound_evidence': str(source_path),
    } for prompt in PROMPTS]
    inputs.extend([
        {'prompt_tokens': cap - 2, 'architectural_output_budget': 2,
         'measured_prefill': True, 'maximum_status': 'success',
         'maximum_output_tokens': 2, 'verified_output_lower_bound': 2,
         'evidence': str(args.output / 'boundary-summary.json'), 'reused': True},
        {'prompt_tokens': cap, 'architectural_output_budget': 0,
         'measured_prefill': False, 'maximum_status': 'architectural-budget-only',
         'maximum_output_tokens': None},
    ])
    measurements = []
    monitor = Path(__file__).resolve().with_name('measure-inference-load.py')
    for phase in ('input', 'output'):
        for prompt in PROMPTS:
            name = f'{phase}-{prompt}'
            output = args.output / name
            generated = 2 if phase == 'input' else cap - prompt
            context = prompt + 2 if phase == 'input' else cap
            command = [sys.executable, str(monitor), str(source_path), str(output),
                       '--trial', f'probe-{prompt}', '--timeout',
                       str(args.input_timeout if phase == 'input' else args.output_timeout)]
            if phase == 'output':
                command.extend(['--sample-len', str(generated), '--max-context', str(cap)])
            measurements.append({
                'name': name, 'phase': phase, 'prompt_tokens': prompt,
                'requested_output_tokens': generated, 'runtime_context': context,
                'status': 'pending', 'command': command,
                'summary_path': str(output / 'summary.json'),
                'log_path': str(args.output / f'{name}.log'),
            })
    summary = {
        'status': 'pending', 'started_utc': utc(), 'runner_pid': os.getpid(),
        'source_summary': str(args.source_summary),
        'source_snapshot': str(source_path),
        'boundary_summary': str(args.boundary_summary),
        'declared_context': cap, 'verified_provenance_sha256': provenance,
        'method': 'Serial synthetic probes. Exact full-budget success establishes the '
                  'maximum under runtime policy, not output quality. Timeout, OOM and '
                  'other failures leave the maximum inconclusive; no bisection. '
                  '262144 input has zero architectural output budget, not measured prefill.',
        'active_trial': None, 'completed_results': [],
        'inputs': inputs, 'trials': measurements,
    }
    summary_path = args.output / 'summary.json'
    save_summary(summary_path, summary)
    for measurement in measurements:
        entry = next(i for i in inputs if i['prompt_tokens'] == measurement['prompt_tokens'])
        measurement.update(status='running', started_utc=utc())
        summary.update(status='running', active_trial=measurement['name'])
        if measurement['phase'] == 'output':
            entry['maximum_status'] = 'running'
        save_summary(summary_path, summary)
        try:
            # The monitor owns GPU-idle checks and inference timeout/cleanup.
            with Path(measurement['log_path']).open('x') as log:
                process = subprocess.run(measurement['command'], stdout=log,
                                         stderr=subprocess.STDOUT, check=False)
            measurement['exit_code'] = process.returncode
            if process.returncode != 0:
                raise ValueError(f'monitor failed with exit code {process.returncode}; see log')
            result = json.loads(Path(measurement['summary_path']).read_text())
            validate_measurement(result, measurement['prompt_tokens'],
                                 measurement['requested_output_tokens'],
                                 measurement['runtime_context'], provenance)
            measurement['status'] = 'success'
            entry['measured_prefill'] = True
            if measurement['phase'] == 'output':
                entry.update(maximum_status='success',
                             maximum_output_tokens=measurement['requested_output_tokens'],
                             verified_output_lower_bound=measurement['requested_output_tokens'],
                             lower_bound_evidence=measurement['summary_path'])
        except Exception as error:
            measurement.update(status='inconclusive', error=f'{type(error).__name__}: {error}')
            if measurement['phase'] == 'output':
                entry['maximum_status'] = 'inconclusive'
        measurement['finished_utc'] = utc()
        summary['completed_results'].append({
            key: measurement[key] for key in
            ('name', 'status', 'summary_path', 'log_path', 'started_utc', 'finished_utc')
        })
        summary['active_trial'] = None
        save_summary(summary_path, summary)
    failed = any(t['status'] != 'success' for t in measurements)
    summary.update(status='inconclusive' if failed else 'success', finished_utc=utc())
    save_summary(summary_path, summary)
    print(summary_path, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
