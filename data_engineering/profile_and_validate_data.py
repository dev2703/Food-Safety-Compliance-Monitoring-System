import pandas as pd
import os
from pathlib import Path

def profile_dataset(file_path, report_path):
    df = pd.read_csv(file_path)
    profile = {
        'file': file_path,
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'sample_rows': df.head(3).to_dict(orient='records'),
        'describe': df.describe(include='all').to_dict()
    }
    with open(report_path, 'w') as f:
        for k, v in profile.items():
            f.write(f'## {k}\n{v}\n\n')
    print(f"Profiled {file_path} -> {report_path}")

def main():
    raw_data_dirs = [
        'data/raw',
        'data/raw/restaurants',
        'data/raw/violations',
        'data/raw/historical'
    ]
    report_dir = Path('data/validation_reports')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    for data_dir in raw_data_dirs:
        for fname in os.listdir(data_dir):
            if fname.endswith('.csv'):
                file_path = os.path.join(data_dir, fname)
                report_path = report_dir / f'{fname}_profile.txt'
                profile_dataset(file_path, report_path)

if __name__ == "__main__":
    main() 