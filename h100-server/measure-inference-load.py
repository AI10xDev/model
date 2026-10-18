#!/usr/bin/env python3
"""Replay or extend a context trial, recording GPU and Linux CPU/RAM load."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def snapshot(pid):
    cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    memory = {line.split(':')[0]: int(line.split()[1])
              for line in Path('/proc/meminfo').read_text().splitlines()}
    result = {
        'monotonic': time.monotonic(),
        'cpu_total_ticks': sum(cpu),
        'cpu_idle_ticks': cpu[3] + cpu[4],
        'host_ram_total_mib': memory['MemTotal'] / 1024,
        'host_ram_used_mib': (memory['MemTotal'] - memory['MemAvailable']) / 1024,
        'host_swap_used_mib': (memory['SwapTotal'] - memory['SwapFree']) / 1024,
        'process_cpu_ticks': None,
        'process_rss_mib': None,
    }
    if pid is not None:
        try:
            # comm is parenthesized and can itself contain spaces or parentheses.
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            result['process_cpu_ticks'] = int(fields[11]) + int(fields[12])
            result['process_rss_mib'] = int(fields[21]) * os.sysconf('SC_PAGE_SIZE') / 1048576
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def cpu_rates(previous, current):
    total = current['cpu_total_ticks'] - previous['cpu_total_ticks']
    idle = current['cpu_idle_ticks'] - previous['cpu_idle_ticks']
    elapsed = current['monotonic'] - previous['monotonic']
    process = None
    if elapsed > 0 and all(s['process_cpu_ticks'] is not None for s in (previous, current)):
        process = 100 * (current['process_cpu_ticks'] - previous['process_cpu_ticks']) / os.sysconf('SC_CLK_TCK') / elapsed
    return {
        'host_cpu_percent': 100 * (total - idle) / total if total > 0 else None,
        'process_cpu_percent_one_core': process,
    }


def aggregate(samples):
    result = {'sample_count': len(samples)}
    for key in ('gpu_util_percent', 'gpu_memory_mib', 'host_cpu_percent',
                'process_cpu_percent_one_core', 'host_ram_used_mib',
                'process_rss_mib', 'host_swap_used_mib'):
        values = [s[key] for s in samples if s.get(key) is not None]
        result[key] = {'min': min(values), 'mean': sum(values) / len(values),
                       'max': max(values)} if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_summary', type=Path)
    parser.add_argument('output', type=Path, help='new directory; never overwrites evidence')
    parser.add_argument('--trial', default='probe-262142')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--sample-len', type=int, help='override requested output tokens (at least 2)')
    parser.add_argument('--max-context', type=int, help='override runtime total token budget')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    source = json.loads(args.source_summary.read_text())
    trial = next(t for t in source['trials'] if t['name'] == args.trial)
    if trial['status'] != 'success':
        parser.error('source trial must have succeeded')
    command = list(trial['command'])
    for flag, value in (('--sample-len', args.sample_len), ('--max-context', args.max_context)):
        if value is not None:
            command[command.index(flag) + 1] = str(value)
    expected = int(command[command.index('--probe-prompt-tokens') + 1])
    generated = int(command[command.index('--sample-len') + 1])
    context = int(command[command.index('--max-context') + 1])
    if generated < 2 or expected <= 0 or expected + generated > context:
        parser.error('require positive input, at least two outputs, and input + output <= max-context')
    if context > trial['declared_context']:
        parser.error('max-context exceeds the source model declared context')
    if '--ignore-eos' not in command:
        parser.error('capacity measurement requires --ignore-eos')
    active = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        parser.error('GPU occupied; refusing to start')
    # Match the actual binary, adapter, config, tokenizer and shards to the prior run.
    for name, expected_hash in source['provenance_sha256'].items():
        if name.endswith('.py'):
            continue
        digest = hashlib.sha256()
        with Path(name).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise SystemExit(f'Provenance mismatch: {name}')
    if subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        parser.error('GPU became occupied during provenance verification; refusing to start')
    args.output.mkdir(exist_ok=False)
    report = args.output / 'inference.json'
    command[command.index('--report-file') + 1] = str(report)
    summary = {
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source_summary': str(args.source_summary), 'source_trial': trial,
        'verified_provenance_sha256': {k: v for k, v in source['provenance_sha256'].items() if not k.endswith('.py')},
        'monitor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'logical_cpus': os.cpu_count(), 'command': command,
        'gpu': subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=name,memory.total,driver_version', '--format=csv'], text=True),
        'conventions': 'Approximately 1-second samples, including model loading. GPU device-wide. Host CPU 0-100% across all CPUs, iowait excluded. Process CPU 100% = one logical CPU. RAM used = MemTotal - MemAvailable; RSS includes resident file mappings. Means are sample means, peaks are sampled, not instantaneous. Final 60s is an observation window, not an exact inference phase.',
        'baseline': snapshot(None),
    }
    (args.output / 'source-summary.json').write_text(json.dumps(source, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/lib64:' + env.get('LD_LIBRARY_PATH', '')
    samples = []
    timed_out = False
    telemetry_error = None
    start = time.monotonic()
    with (args.output / 'inference.log').open('w') as log, (args.output / 'telemetry.jsonl').open('w', buffering=1) as telemetry:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            previous = snapshot(process.pid)
            while process.poll() is None:
                gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10)
                current = snapshot(process.pid)
                sample = dict(current, **cpu_rates(previous, current))
                sample['elapsed_seconds'] = current['monotonic'] - start
                sample['gpu_util_percent'], sample['gpu_memory_mib'] = map(float, gpu.strip().split(','))
                samples.append(sample)
                telemetry.write(json.dumps(sample) + '\n')
                previous = current
                if time.monotonic() - start > args.timeout:
                    timed_out = True
                    break
                time.sleep(1)
        except Exception as error:
            telemetry_error = f'{type(error).__name__}: {error}'
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    summary.update(exit_code=process.returncode, timed_out=timed_out,
                   telemetry_error=telemetry_error,
                   wall_seconds=time.monotonic() - start,
                   finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   whole_process=aggregate(samples),
                   final_60_seconds=aggregate([s for s in samples if s['elapsed_seconds'] >= samples[-1]['elapsed_seconds'] - 60]))
    try:
        result = json.loads(report.read_text()) if report.exists() else {}
    except (ValueError, OSError) as error:
        result = {'report_error': str(error)}
    summary['inference'] = result
    summary['status'] = 'success' if not timed_out and telemetry_error is None and process.returncode == 0 and all([
        result.get('status') == 'success', result.get('prompt_tokens') == expected,
        result.get('generated_tokens') == generated,
        result.get('forwarded_tokens') == expected + generated - 1,
        result.get('max_forwarded_position_zero_based') == expected + generated - 2,
        result.get('runtime_context') == context, result.get('cached_decode_tested') is True,
    ]) else 'failed'
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if summary['status'] != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
