import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

def generate_restaurant_data(num_restaurants=1000):
    """Generate simulated restaurant data."""
    restaurant_types = ['Restaurant', 'Cafe', 'Food Truck', 'Bakery', 'Grocery Store']
    risk_levels = ['Risk 1 (High)', 'Risk 2 (Medium)', 'Risk 3 (Low)']
    
    data = {
        'Restaurant ID': range(1, num_restaurants + 1),
        'Name': [f'Restaurant_{i}' for i in range(1, num_restaurants + 1)],
        'Type': np.random.choice(restaurant_types, num_restaurants),
        'Risk Level': np.random.choice(risk_levels, num_restaurants),
        'Address': [f'{random.randint(1, 9999)} Main St' for _ in range(num_restaurants)],
        'City': ['Chicago'] * num_restaurants,
        'State': ['IL'] * num_restaurants,
        'Zip': [f'606{random.randint(1, 99):02d}' for _ in range(num_restaurants)],
        'License Number': [f'L{random.randint(1000000, 9999999)}' for _ in range(num_restaurants)],
        'Opening Date': [(datetime.now() - timedelta(days=random.randint(1, 3650))).strftime('%Y-%m-%d') 
                        for _ in range(num_restaurants)],
        'Capacity': np.random.randint(20, 200, num_restaurants),
        'Has Delivery': np.random.choice([True, False], num_restaurants),
        'Has Outdoor Seating': np.random.choice([True, False], num_restaurants)
    }
    
    return pd.DataFrame(data)

def generate_violations_data(num_violations=5000):
    """Generate simulated violation data."""
    violation_types = [
        'Food Temperature', 'Sanitation', 'Pest Control', 'Equipment Maintenance',
        'Employee Hygiene', 'Food Storage', 'Cross Contamination', 'Documentation'
    ]
    severity_levels = ['Critical', 'Serious', 'Minor']
    
    data = {
        'Violation ID': range(1, num_violations + 1),
        'Restaurant ID': np.random.randint(1, 1001, num_violations),
        'Inspection ID': np.random.randint(1, 10001, num_violations),
        'Violation Type': np.random.choice(violation_types, num_violations),
        'Severity': np.random.choice(severity_levels, num_violations),
        'Description': [f'Violation description for {i}' for i in range(1, num_violations + 1)],
        'Date': [(datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d') 
                for _ in range(num_violations)],
        'Corrected': np.random.choice([True, False], num_violations),
        'Correction Date': [(datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d') 
                           if random.random() > 0.3 else None for _ in range(num_violations)]
    }
    
    return pd.DataFrame(data)

def generate_historical_inspections(num_inspections=10000):
    """Generate simulated historical inspection data."""
    inspection_types = ['Routine', 'Complaint', 'Follow-up', 'License']
    results = ['Pass', 'Fail', 'Pass w/ Conditions']
    
    data = {
        'Inspection ID': range(1, num_inspections + 1),
        'Restaurant ID': np.random.randint(1, 1001, num_inspections),
        'Inspection Type': np.random.choice(inspection_types, num_inspections),
        'Date': [(datetime.now() - timedelta(days=random.randint(1, 3650))).strftime('%Y-%m-%d') 
                for _ in range(num_inspections)],
        'Result': np.random.choice(results, num_inspections),
        'Score': np.random.randint(0, 101, num_inspections),
        'Inspector ID': [f'INSP{random.randint(1000, 9999)}' for _ in range(num_inspections)],
        'Notes': [f'Inspection notes for inspection {i}' for i in range(1, num_inspections + 1)]
    }
    
    return pd.DataFrame(data)

def main():
    # Create output directories if they don't exist
    output_dirs = ['data/raw/restaurants', 'data/raw/violations', 'data/raw/historical']
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    restaurants_df = generate_restaurant_data()
    violations_df = generate_violations_data()
    historical_df = generate_historical_inspections()
    
    # Save datasets
    restaurants_df.to_csv('data/raw/restaurants/restaurants.csv', index=False)
    violations_df.to_csv('data/raw/violations/violations.csv', index=False)
    historical_df.to_csv('data/raw/historical/historical_inspections.csv', index=False)
    
    print("Simulated datasets have been generated successfully!")

if __name__ == "__main__":
    main() 