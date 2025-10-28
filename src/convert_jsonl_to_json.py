#!/usr/bin/env python3
"""
Converts a JSON Lines (.jsonl) file to a standard JSON array file.
"""
import json
import argparse
import os

def convert_jsonl_to_json(input_path, output_path):
    """
    Reads a .jsonl file, converts it to a list of Python dictionaries,
    and saves it as a standard, pretty-printed .json file.

    Args:
        input_path (str): Path to the input .jsonl file.
        output_path (str): Path to the output .json file.
    """
    print(f"🚀 Starting conversion...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at '{input_path}'")
        return

    # Read the .jsonl file
    data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error reading or parsing {input_path}: {e}")
        return

    print(f"✅ Successfully loaded {len(data):,} records from {input_path}.")

    # Write the .json file
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error writing to {output_path}: {e}")
        return

    print(f"✅ Successfully converted and saved to {output_path}.")
    print("✨ Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a JSON Lines (.jsonl) file to a standard JSON array file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Path to the input .jsonl file."
    )
    parser.add_argument(
        "output_file",
        help="Path for the output .json file."
    )

    args = parser.parse_args()

    convert_jsonl_to_json(args.input_file, args.output_file)
