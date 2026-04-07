"""Parse experiment logs and generate a CSV summary.

Scans logs/*.log for lines matching:
    RESULT|<setting>|<split>|mse=...|mae=...|rmse=...|mape=...|mspe=...

Setting format: {model}_{data}_{features}_sl{seq_len}_pl{pred_len}

Usage:
    python scripts/analysis/parse_logs.py                    # default
    python scripts/analysis/parse_logs.py --log_dir path/    # custom log dir
    python scripts/analysis/parse_logs.py -o results.csv     # custom output
"""

import argparse
import csv
import re
from pathlib import Path


RESULT_PATTERN = re.compile(
    r'^RESULT\|'
    r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|'
    r'(?P<setting>[^|]+)\|'
    r'(?P<split>[^|]+)\|'
    r'mse=(?P<mse>[\d.]+)\|'
    r'mae=(?P<mae>[\d.]+)'
    r'(?:\|rmse=(?P<rmse>[\d.]+))?'
    r'(?:\|mape=(?P<mape>[\d.]+))?'
    r'(?:\|mspe=(?P<mspe>[\d.]+))?$'
)

SETTING_PATTERN = re.compile(
    r'^(?P<model>[^_]+)_(?P<data>[^_]+)_(?P<features>[^_]+)_sl(?P<seq_len>\d+)_pl(?P<pred_len>\d+)$'
)

COLUMNS = ['timestamp', 'model', 'data', 'features', 'seq_len', 'pred_len', 'split', 'mse', 'mae', 'rmse', 'mape', 'mspe']
METRICS = ['mse', 'mae', 'rmse', 'mape', 'mspe']


def parse_log(filepath: Path) -> list[dict]:
    """Extract all RESULT lines from a single log file."""
    rows = []
    for line in filepath.read_text().splitlines():
        m = RESULT_PATTERN.match(line.strip())
        if not m:
            continue
        sm = SETTING_PATTERN.match(m.group('setting'))
        if not sm:
            continue
        row = {
            'timestamp': m.group('timestamp'),
            'model': sm.group('model'),
            'data': sm.group('data'),
            'features': sm.group('features'),
            'seq_len': int(sm.group('seq_len')),
            'pred_len': int(sm.group('pred_len')),
            'split': m.group('split'),
        }
        for met in METRICS:
            val = m.group(met)
            row[met] = float(val) if val else ''
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Parse experiment logs to CSV')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory containing .log files')
    parser.add_argument('-o', '--output', type=str, default='outputs/results_summary.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f'Log directory not found: {log_dir}')
        return

    log_files = sorted(log_dir.glob('*.log'))
    if not log_files:
        print(f'No .log files in {log_dir}')
        return

    all_rows = []
    for f in log_files:
        rows = parse_log(f)
        all_rows.extend(rows)
        if rows:
            print(f'  {f.name}: {len(rows)} result(s)')
        else:
            print(f'  {f.name}: no RESULT lines found')

    # Sort: model -> data -> pred_len
    all_rows.sort(key=lambda r: (r['model'], r['data'], r['pred_len'], r['split'], r['timestamp']))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\n{len(all_rows)} results written to {output_path}')


if __name__ == '__main__':
    main()
