import os
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create data directories if they don't exist
DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs and configurations
DATASETS = {
    'nyc': {
        'url': 'https://data.cityofnewyork.us/api/views/43nn-pn8j/rows.csv',
        'filename': 'nyc_inspections.csv',
        'description': 'NYC Restaurant Inspection Results'
    },
    'chicago': {
        'url': 'https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv',
        'filename': 'chicago_inspections.csv',
        'description': 'Chicago Food Inspections'
    },
    'usda': {
        'url': 'https://www.fsis.usda.gov/sites/default/files/media_file/2023-12/FSIS_Establishment_Data.csv',
        'filename': 'usda_inspections.csv',
        'description': 'USDA Establishment Data'
    }
}

def download_file(url: str, filename: str) -> bool:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        logger.error(f"Error downloading {filename}: {str(e)}")
        return False

def validate_dataset(df: pd.DataFrame, dataset_name: str) -> bool:
    """
    Perform basic validation on downloaded dataset.
    
    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset for logging
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            logger.error(f"{dataset_name} dataset is empty")
            return False
            
        # Check for minimum required columns based on dataset
        if dataset_name == 'nyc':
            required_cols = ['CAMIS', 'DBA', 'BORO', 'GRADE', 'SCORE']
        elif dataset_name == 'chicago':
            required_cols = ['Inspection_ID', 'DBA_Name', 'Risk', 'Results']
        elif dataset_name == 'usda':
            required_cols = ['Establishment Number', 'Company', 'City', 'State']
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"{dataset_name} missing required columns: {missing_cols}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating {dataset_name}: {str(e)}")
        return False

def main():
    """Main function to download and validate all datasets."""
    logger.info("Starting data acquisition process...")
    
    for dataset_name, config in DATASETS.items():
        logger.info(f"Processing {config['description']}...")
        
        # Download file
        output_path = RAW_DIR / config['filename']
        if download_file(config['url'], output_path):
            logger.info(f"Successfully downloaded {config['filename']}")
            
            # Read and validate dataset
            try:
                df = pd.read_csv(output_path)
                if validate_dataset(df, dataset_name):
                    logger.info(f"Successfully validated {config['filename']}")
                    
                    # Save basic statistics
                    stats_path = RAW_DIR / f"{dataset_name}_stats.txt"
                    with open(stats_path, 'w') as f:
                        f.write(f"Dataset: {config['description']}\n")
                        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Number of Records: {len(df)}\n")
                        f.write(f"Number of Columns: {len(df.columns)}\n")
                        f.write("\nColumn Names:\n")
                        f.write("\n".join(df.columns))
                        f.write("\n\nSample Data:\n")
                        f.write(df.head().to_string())
                else:
                    logger.error(f"Validation failed for {config['filename']}")
            except Exception as e:
                logger.error(f"Error processing {config['filename']}: {str(e)}")
        else:
            logger.error(f"Failed to download {config['filename']}")
    
    logger.info("Data acquisition process completed.")

if __name__ == "__main__":
    main() 