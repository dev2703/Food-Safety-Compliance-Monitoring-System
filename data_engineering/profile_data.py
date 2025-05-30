import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sweetviz as sv
from pandas_profiling import ProfileReport
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'

# Create necessary directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load a dataset from the raw directory.
    
    Args:
        filename: Name of the file to load
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        file_path = RAW_DIR / filename
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {filename}")
        return df
    except Exception as e:
        logger.error(f"Error loading {filename}: {str(e)}")
        raise

def generate_pandas_profile(df: pd.DataFrame, dataset_name: str):
    """
    Generate a pandas profiling report.
    
    Args:
        df: DataFrame to profile
        dataset_name: Name of the dataset
    """
    try:
        profile = ProfileReport(
            df,
            title=f"{dataset_name} Profiling Report",
            explorative=True
        )
        
        output_path = REPORTS_DIR / f"{dataset_name}_pandas_profile.html"
        profile.to_file(output_path)
        logger.info(f"Generated pandas profile for {dataset_name}")
    except Exception as e:
        logger.error(f"Error generating pandas profile for {dataset_name}: {str(e)}")

def generate_sweetviz_report(df: pd.DataFrame, dataset_name: str):
    """
    Generate a Sweetviz report.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset
    """
    try:
        report = sv.analyze(df)
        output_path = REPORTS_DIR / f"{dataset_name}_sweetviz.html"
        report.show_html(output_path)
        logger.info(f"Generated Sweetviz report for {dataset_name}")
    except Exception as e:
        logger.error(f"Error generating Sweetviz report for {dataset_name}: {str(e)}")

def analyze_dataset(df: pd.DataFrame, dataset_name: str):
    """
    Perform comprehensive analysis on a dataset.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset
    """
    try:
        # Basic statistics
        stats = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
            'categorical_stats': {
                col: df[col].value_counts().to_dict()
                for col in df.select_dtypes(include=['object']).columns
            }
        }
        
        # Save statistics
        stats_path = REPORTS_DIR / f"{dataset_name}_analysis.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved analysis for {dataset_name}")
        
        # Generate reports
        generate_pandas_profile(df, dataset_name)
        generate_sweetviz_report(df, dataset_name)
        
    except Exception as e:
        logger.error(f"Error analyzing {dataset_name}: {str(e)}")

def main():
    """Main function to profile all datasets."""
    try:
        logger.info("Starting data profiling...")
        
        # Process each dataset
        datasets = {
            'nyc': 'nyc_inspections.csv',
            'chicago': 'chicago_inspections.csv',
            'usda': 'usda_inspections.csv'
        }
        
        for dataset_name, filename in datasets.items():
            logger.info(f"Processing {dataset_name}...")
            df = load_dataset(filename)
            analyze_dataset(df, dataset_name)
            
        logger.info("Data profiling completed successfully")
        
    except Exception as e:
        logger.error(f"Data profiling failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 