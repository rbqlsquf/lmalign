#!/usr/bin/env python3
"""Convert metrics.json to CSV format for model comparison.
Each row is a model, columns are benchmark metrics.
"""

import json
import sys
import csv
import re
from pathlib import Path


def extract_checkpoint_number(model_name):
    """Extract checkpoint number from model name for sorting.

    Args:
        model_name: Model name like 'checkpoint-500', 'checkpoint-1000', etc.

    Returns:
        Tuple (number, original_name) for sorting. If no number found, returns (float('inf'), model_name)
    """
    if not model_name:
        return (float('inf'), model_name)

    # Try to extract number from checkpoint-XXX pattern
    match = re.search(r'checkpoint-(\d+)', model_name, re.IGNORECASE)
    if match:
        return (int(match.group(1)), model_name)

    # If no checkpoint pattern found, sort by name
    return (float('inf'), model_name)


def get_important_metrics(benchmark):
    """Return list of important metrics for a given benchmark."""
    benchmark_lower = benchmark.lower()

    # Math benchmarks (gsm8k, math, etc.) - symbolic_correct is the main metric
    if 'gsm8k' in benchmark_lower or 'math' in benchmark_lower or benchmark_lower.startswith('mmlu'):
        return ['symbolic_correct']

    # LiveCodeBench - accuracy is the main metric
    if 'livecodebench' in benchmark_lower:
        return ['accuracy']

    # Code benchmarks (human-eval, etc.) - passing tests are the main metrics
    if 'human-eval' in benchmark_lower or 'humaneval' in benchmark_lower:
        return ['passing_base_tests', 'passing_plus_tests']

    # IFEval - average_score and instruction_strict_accuracy are main metrics
    if 'ifeval' in benchmark_lower:
        return ['average_score', 'instruction_strict_accuracy']

    # Default: try to find common important metrics
    # This will be filtered based on what's actually available
    return ['symbolic_correct', 'passing_base_tests', 'passing_plus_tests',
            'average_score', 'instruction_strict_accuracy', 'judge_correct']


def extract_metrics_from_json(json_path, model_name=None):
    """Extract metrics from a single JSON file and return as a dictionary.

    Args:
        json_path: Path to metrics.json file
        model_name: Name of the model (if None, will try to extract from path)

    Returns:
        Dictionary with model name and metrics
    """
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    # Extract model name from path if not provided
    if model_name is None:
        # Path structure: .../sft/checkpoint-1000/metrics.json
        # parent directory is the model/checkpoint name
        parent = Path(json_path).parent
        model_name = parent.name

    # Build column names: benchmark_metric_name
    column_data = {}
    column_data['model'] = model_name

    for benchmark, eval_modes in metrics.items():
        important_metrics = get_important_metrics(benchmark)

        for eval_mode, metric_values in eval_modes.items():
            # Only process pass@1 for now (can be extended)
            if eval_mode != 'pass@1':
                continue

            # Add important metrics that exist in the data
            found_important = False
            for metric_name in important_metrics:
                if metric_name in metric_values:
                    column_name = f"{benchmark}_{metric_name}"
                    column_data[column_name] = metric_values[metric_name]
                    found_important = True

            # If no important metrics were found, include the first numeric metric as fallback
            if not found_important:
                for metric_name, metric_value in metric_values.items():
                    if isinstance(metric_value, (int, float)) and metric_name not in ['num_entries', 'avg_tokens', 'gen_seconds']:
                        column_name = f"{benchmark}_{metric_name}"
                        column_data[column_name] = metric_value
                        break

    return column_data


def json_to_csv(json_path, csv_path, model_name=None):
    """Convert metrics.json to CSV format for model comparison.

    Args:
        json_path: Path to metrics.json file
        csv_path: Path to output CSV file
        model_name: Name of the model (if None, will try to extract from path)
    """
    column_data = extract_metrics_from_json(json_path, model_name)

    # Create CSV with just this model's data
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort columns: model first, then benchmark metrics
    all_columns = ['model'] + sorted([k for k in column_data.keys() if k != 'model'])

    # Write CSV with single row
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerow(column_data)

    print(f"CSV saved to {csv_path}")


def batch_process_directory(directory_path, output_csv_path):
    """Process all summary/metrics.json files in a directory and combine into one CSV.

    Args:
        directory_path: Root directory to search for metrics.json files
        output_csv_path: Path to output CSV file with all models
    """
    directory_path = Path(directory_path)
    if not directory_path.exists():
        print(f"Error: Directory not found: {directory_path}")
        sys.exit(1)

    # Find all metrics.json files directly under checkpoint directories
    # e.g., sft/checkpoint-1000/metrics.json
    metrics_files = list(directory_path.glob('*/metrics.json'))

    if not metrics_files:
        print(f"Error: No */metrics.json files found in {directory_path}")
        sys.exit(1)

    print(f"Found {len(metrics_files)} metrics.json files")

    # Process each file
    all_rows = []
    all_columns_set = set(['model'])

    for json_path in sorted(metrics_files):
        print(f"Processing: {json_path}")
        try:
            column_data = extract_metrics_from_json(json_path)
            all_rows.append(column_data)
            all_columns_set.update(column_data.keys())
        except Exception as e:
            print(f"Warning: Failed to process {json_path}: {e}")
            continue

    if not all_rows:
        print("Error: No valid metrics found")
        sys.exit(1)

    # Sort rows by checkpoint number (extracted from model name)
    all_rows.sort(key=lambda row: extract_checkpoint_number(row.get('model', '')))

    # Sort columns: model first, then benchmark metrics
    all_columns = ['model'] + sorted([k for k in all_columns_set if k != 'model'])

    # Write combined CSV
    output_csv_path = Path(output_csv_path)

    # If output path is a directory or has no extension, handle it appropriately
    if output_csv_path.exists() and output_csv_path.is_dir():
        # If it's an existing directory, add default filename
        output_csv_path = output_csv_path / 'combined_metrics.csv'
    elif not output_csv_path.suffix:
        # If no extension and doesn't exist, add .csv extension
        output_csv_path = Path(str(output_csv_path) + '.csv')

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for row in all_rows:
            # Fill missing columns with empty values
            complete_row = {col: row.get(col, '') for col in all_columns}
            writer.writerow(complete_row)

    print(f"\nCombined CSV saved to {output_csv_path}")
    print(f"Total models: {len(all_rows)}")


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(f"Usage:")
        print(f"  Single file: {sys.argv[0]} <json_path> <csv_path> [model_name]")
        print(f"  Batch directory: {sys.argv[0]} --batch <directory_path> <output_csv_path>")
        print("")
        print("Examples:")
        print(f"  {sys.argv[0]} metrics.json output.csv")
        print(f"  {sys.argv[0]} --batch /path/to/eval/sft combined_results.csv")
        sys.exit(1)

    # Check if batch mode
    if sys.argv[1] == '--batch':
        if len(sys.argv) != 4:
            print(f"Usage: {sys.argv[0]} --batch <directory_path> <output_csv_path>")
            sys.exit(1)
        directory_path = Path(sys.argv[2])
        output_csv_path = Path(sys.argv[3])
        batch_process_directory(directory_path, output_csv_path)
    else:
        # Single file mode
        json_path = Path(sys.argv[1])
        csv_path = Path(sys.argv[2])
        model_name = sys.argv[3] if len(sys.argv) == 4 else None

        if not json_path.exists():
            print(f"Error: JSON file not found: {json_path}")
            sys.exit(1)

        json_to_csv(json_path, csv_path, model_name)

