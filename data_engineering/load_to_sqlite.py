import pandas as pd
import sqlite3
from pathlib import Path
import os

def load_csv_to_sqlite(csv_path, table_name, conn):
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Loaded {csv_path} into table '{table_name}'")

def main():
    db_path = 'food_safety.db'
    conn = sqlite3.connect(db_path)
    
    # Map CSVs to table names
    csv_table_map = {
        'data/raw/Food_Inspections_20250530.csv': 'food_inspections',
        'data/raw/restaurants/restaurants.csv': 'restaurants',
        'data/raw/violations/violations.csv': 'violations',
        'data/raw/historical/historical_inspections.csv': 'historical_inspections',
    }
    
    for csv_path, table_name in csv_table_map.items():
        if Path(csv_path).exists():
            load_csv_to_sqlite(csv_path, table_name, conn)
        else:
            print(f"Warning: {csv_path} not found, skipping.")
    
    conn.close()
    print(f"All data loaded into {db_path}.")

if __name__ == "__main__":
    main() 