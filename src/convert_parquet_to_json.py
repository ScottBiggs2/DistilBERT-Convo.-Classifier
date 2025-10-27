import pandas as pd
import pyarrow.parquet as pq
import json
import numpy as np

def convert_parquet_to_json(parquet_path: str, json_path: str):
    """
    Convert a Parquet file to JSON format with proper structure.
    
    Args:
        parquet_path (str): Path to the input Parquet file.
        json_path (str): Path to the output JSON file.
    """
    try:
        # Read the Parquet file
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        print(f"Successfully read Parquet file: {parquet_path}")
        print(f"Number of records: {len(df)}")
        
        # Write to JSON file with streaming
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write('[\n')  # Start the JSON array
            
            for idx, row in df.iterrows():
                # Handle messages field - convert numpy array or list to proper format
                messages = row['messages']
                if isinstance(messages, np.ndarray):
                    messages = messages.tolist()
                elif isinstance(messages, str):
                    messages = json.loads(messages)
                
                record = {
                    "chat_id": row['chat_id'],
                    "context": row['context'],
                    "query": row['query'],
                    "messages": messages,
                    "formatted_chat": row['formatted_chat']
                }
                
                # Write the record
                json_str = json.dumps(record, indent=2, ensure_ascii=False)
                
                # Indent the record content to align with array structure
                indented_json = '\n'.join('  ' + line for line in json_str.split('\n'))
                f.write(indented_json)
                
                # Add comma if not the last record
                if idx < len(df) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
                
                # Progress indicator
                if (idx + 1) % 10000 == 0:
                    print(f"Processed {idx + 1} records...")
            
            f.write(']')  # Close the JSON array
        
        print(f"Successfully wrote JSON file: {json_path}")
        print(f"Total records converted: {len(df)}")
        
    except Exception as e:
        print(f"Error converting Parquet to JSON: {e}")
        import traceback
        traceback.print_exc()

def main():
    parquet_path = "data/100k_convo_Oct_2025.parquet"
    json_path = "data/100k_convo_Oct_2025.json"
    
    convert_parquet_to_json(parquet_path, json_path)

if __name__ == "__main__":
    main()