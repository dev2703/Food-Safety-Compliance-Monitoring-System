import pandas as pd
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DB_PATH = 'food_safety.db'
EDA_DIR = Path('eda_reports')
EDA_DIR.mkdir(exist_ok=True)

# Utility: Save plot
def save_plot(fig, name):
    fig.savefig(EDA_DIR / name, bbox_inches='tight')
    plt.close(fig)

def load_table(table_name):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f'SELECT * FROM {table_name}', conn)

def clean_food_inspections(df):
    # Standardize column names
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    # Parse dates
    if 'inspection_date' in df.columns:
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    # Remove rows with no inspection id
    if 'inspection_id' in df.columns:
        df = df[df['inspection_id'].notnull()]
    # Fill missing results as 'Unknown'
    if 'results' in df.columns:
        df['results'] = df['results'].fillna('Unknown')
    return df

def feature_engineer(df):
    # Example: Days since last inspection
    if 'inspection_date' in df.columns:
        df = df.sort_values(['dba_name','inspection_date'])
        df['days_since_last'] = df.groupby('dba_name')['inspection_date'].diff().dt.days
    # Example: Violation count
    if 'violations' in df.columns:
        df['violation_count'] = df['violations'].fillna('').apply(lambda x: len(str(x).split('|')) if x else 0)
    return df

def eda_summary(df, name):
    # Save describe
    desc = df.describe(include='all')
    desc.to_csv(EDA_DIR / f'{name}_describe.csv')
    # Plot result distribution
    if 'results' in df.columns:
        fig, ax = plt.subplots()
        df['results'].value_counts().plot(kind='bar', ax=ax, title='Inspection Results')
        save_plot(fig, f'{name}_results_dist.png')
    # Plot violation count
    if 'violation_count' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['violation_count'], bins=20, ax=ax)
        ax.set_title('Violation Count Distribution')
        save_plot(fig, f'{name}_violation_count_dist.png')

def clean_restaurants(df):
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    if 'opening_date' in df.columns:
        df['opening_date'] = pd.to_datetime(df['opening_date'], errors='coerce')
    if 'capacity' in df.columns:
        df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
    return df

def feature_engineer_restaurants(df):
    # Example: Restaurant age in years
    if 'opening_date' in df.columns:
        df['restaurant_age_years'] = (pd.Timestamp('now') - df['opening_date']).dt.days // 365
    return df

def clean_violations(df):
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def feature_engineer_violations(df):
    # Example: Days to correction
    if 'date' in df.columns and 'correction_date' in df.columns:
        df['correction_date'] = pd.to_datetime(df['correction_date'], errors='coerce')
        df['days_to_correction'] = (df['correction_date'] - df['date']).dt.days
    return df

def clean_historical(df):
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
    return df

def feature_engineer_historical(df):
    # Example: Pass rate by restaurant
    if 'result' in df.columns:
        df['pass'] = df['result'].str.contains('Pass', case=False, na=False)
    return df

def main():
    # Food Inspections
    inspections = clean_food_inspections(load_table('food_inspections'))
    inspections = feature_engineer(inspections)
    eda_summary(inspections, 'food_inspections')
    inspections.to_csv(EDA_DIR / 'food_inspections_cleaned.csv', index=False)

    # Restaurants
    restaurants = clean_restaurants(load_table('restaurants'))
    restaurants = feature_engineer_restaurants(restaurants)
    eda_summary(restaurants, 'restaurants')
    restaurants.to_csv(EDA_DIR / 'restaurants_cleaned.csv', index=False)

    # Violations
    violations = clean_violations(load_table('violations'))
    violations = feature_engineer_violations(violations)
    eda_summary(violations, 'violations')
    violations.to_csv(EDA_DIR / 'violations_cleaned.csv', index=False)

    # Historical Inspections
    historical = clean_historical(load_table('historical_inspections'))
    historical = feature_engineer_historical(historical)
    eda_summary(historical, 'historical_inspections')
    historical.to_csv(EDA_DIR / 'historical_inspections_cleaned.csv', index=False)

    print('EDA and cleaning complete for all datasets. Results in eda_reports/.')

if __name__ == "__main__":
    main() 