import pandas as pd
import pyarrow.parquet as pq

def check_parquet_file(file_path: str) -> pd.DataFrame:
    """
    Check if a Parquet file can be read and return its contents as a DataFrame.
    
    Args:
        file_path (str): Path to the Parquet file.
    
    """
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        print(f"Successfully read Parquet file: {file_path}")
        return df
    except Exception as e:
        print(f"Error reading Parquet file: {file_path}. Error: {e}")
        return pd.DataFrame()
    
def main(): 
    file_path = "data/100k_convo_Oct_2025.parquet"
    df = check_parquet_file(file_path)
    if not df.empty:
        print(df.head())
        print(df.info())

if __name__ == "__main__":
    main()

