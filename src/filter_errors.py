#!/usr/bin/env python3
"""
Filters a JSON file to remove records with processing errors.
"""
import json
import argparse
import os

def filter_error_records(input_path: str, output_path: str, field: str = "intent", error_value: str = "unclear_error"):
    """
    Loads a JSON file containing a list of records, removes records where
    a specific field matches an error value, and saves the clean data.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path for the output filtered JSON file.
        field (str): The key to check for the error value.
        error_value (str): The value that indicates an error record.
    """
    print("🚀 Starting error filtering process...")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  Filtering where '{field}' == '{error_value}'")

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at '{input_path}'")
        return

    # Load the source JSON file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading or parsing {input_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"❌ Error: Input JSON must be an array of objects, but found type '{type(data).__name__}'.")
        return

    initial_count = len(data)
    print(f"✅ Loaded {initial_count:,} records.")

    # Filter out records with the specified error
    clean_data = [
        record for record in data
        if record.get(field) != error_value
    ]

    final_count = len(clean_data)
    removed_count = initial_count - final_count

    print(f"🔍 Removed {removed_count:,} records with errors.")
    print(f"💾 Saving {final_count:,} clean records to {output_path}...")

    # Write the filtered data to the output file
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error writing to {output_path}: {e}")
        return

    print("✨ Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter records from a JSON file based on a specific error value.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSON file (must be a JSON array)."
    )
    parser.add_argument(
        "output_file",
        help="Path for the output filtered JSON file."
    )
    parser.add_argument(
        "--field",
        default="intent",
        help="The JSON field to check for the error value (default: 'intent')."
    )
    parser.add_argument(
        "--error_value",
        default="unclear_error",
        help="The value indicating an error that should be filtered out (default: 'unclear_error')."
    )

    args = parser.parse_args()

    filter_error_records(args.input_file, args.output_file, args.field, args.error_value)
