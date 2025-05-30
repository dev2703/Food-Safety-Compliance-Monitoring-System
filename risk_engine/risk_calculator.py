import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional

class RiskCalculator:
    def __init__(self, db_path: str = 'food_safety.db'):
        """Initialize the risk calculator with database connection."""
        self.db_path = db_path
        self.weights = {
            'violation_history': 0.3,
            'inspection_frequency': 0.2,
            'violation_severity': 0.25,
            'time_since_last': 0.15,
            'establishment_type': 0.1
        }

    def _get_establishment_data(self, dba_name: str) -> tuple:
        """Fetch all relevant data for a restaurant."""
        with sqlite3.connect(self.db_path) as conn:
            # Get inspection history
            inspections = pd.read_sql(
                f"SELECT * FROM food_inspections WHERE dba_name = '{dba_name}'",
                conn
            )
            
            # Get violations from the inspections
            violations = []
            if not inspections.empty:
                for _, row in inspections.iterrows():
                    if pd.notna(row['violations']):
                        violations.extend(row['violations'].split('|'))
            
            return inspections, violations

    def calculate_violation_score(self, violations: List[str]) -> float:
        """Calculate risk score based on violation history."""
        if not violations:
            return 0.0
        
        # Count critical violations
        critical_count = sum(1 for v in violations if 'CRITICAL' in v.upper())
        serious_count = sum(1 for v in violations if 'SERIOUS' in v.upper())
        minor_count = sum(1 for v in violations if 'MINOR' in v.upper())
        
        # Weight violations
        total_weighted = (critical_count * 1.0 + serious_count * 0.7 + minor_count * 0.3)
        total_violations = len(violations)
        
        return min(1.0, total_weighted / max(1, total_violations))

    def calculate_inspection_frequency_score(self, inspections: pd.DataFrame) -> float:
        """Calculate risk score based on inspection frequency."""
        if inspections.empty:
            return 1.0  # High risk if no inspections
        
        # Calculate average days between inspections
        inspections['inspection_date'] = pd.to_datetime(inspections['inspection_date'])
        inspections = inspections.sort_values('inspection_date')
        days_between = inspections['inspection_date'].diff().dt.days
        
        if days_between.empty:
            return 0.5
        
        avg_days = days_between.mean()
        
        # Score based on frequency (lower is better)
        if avg_days <= 30:
            return 0.2
        elif avg_days <= 90:
            return 0.5
        elif avg_days <= 180:
            return 0.7
        else:
            return 1.0

    def calculate_time_since_last_score(self, inspections: pd.DataFrame) -> float:
        """Calculate risk score based on time since last inspection."""
        if inspections.empty:
            return 1.0
        
        last_inspection = pd.to_datetime(inspections['inspection_date']).max()
        days_since = (datetime.now() - last_inspection).days
        
        # Score based on time since last inspection
        if days_since <= 30:
            return 0.2
        elif days_since <= 90:
            return 0.5
        elif days_since <= 180:
            return 0.7
        else:
            return 1.0

    def calculate_establishment_type_score(self, facility_type: str) -> float:
        """Calculate risk score based on establishment type."""
        type_risk = {
            'Restaurant': 0.8,
            'Cafe': 0.6,
            'Food Truck': 0.7,
            'Bakery': 0.5,
            'Grocery Store': 0.4
        }
        
        return type_risk.get(facility_type, 0.5)

    def calculate_risk_score(self, dba_name: str) -> Dict[str, float]:
        """Calculate overall risk score for a restaurant."""
        inspections, violations = self._get_establishment_data(dba_name)
        
        if inspections.empty:
            return {
                'total_score': 1.0,
                'components': {
                    'violation_score': 1.0,
                    'frequency_score': 1.0,
                    'time_score': 1.0,
                    'type_score': 0.5
                }
            }
        
        # Get facility type from first inspection
        facility_type = inspections['facility_type'].iloc[0]
        
        # Calculate individual component scores
        violation_score = self.calculate_violation_score(violations)
        frequency_score = self.calculate_inspection_frequency_score(inspections)
        time_score = self.calculate_time_since_last_score(inspections)
        type_score = self.calculate_establishment_type_score(facility_type)
        
        # Calculate weighted total score
        total_score = (
            self.weights['violation_history'] * violation_score +
            self.weights['inspection_frequency'] * frequency_score +
            self.weights['time_since_last'] * time_score +
            self.weights['establishment_type'] * type_score
        )
        
        return {
            'total_score': total_score,
            'components': {
                'violation_score': violation_score,
                'frequency_score': frequency_score,
                'time_score': time_score,
                'type_score': type_score
            }
        }

    def get_risk_level(self, score: float) -> str:
        """Convert numerical risk score to risk level."""
        if score >= 0.8:
            return 'Critical'
        elif score >= 0.6:
            return 'High'
        elif score >= 0.4:
            return 'Medium'
        else:
            return 'Low'

    def update_restaurant_risk(self, dba_name: str) -> None:
        """Update risk score in the database."""
        risk_data = self.calculate_risk_score(dba_name)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE food_inspections 
                SET risk_score = ?, 
                    risk_level = ?,
                    last_risk_update = ?
                WHERE dba_name = ?
            """, (
                risk_data['total_score'],
                self.get_risk_level(risk_data['total_score']),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                dba_name
            ))
            conn.commit()

def main():
    """Example usage of the risk calculator."""
    calculator = RiskCalculator()
    
    # Get a sample restaurant name from the database
    with sqlite3.connect('food_safety.db') as conn:
        sample_restaurant = pd.read_sql(
            "SELECT DISTINCT dba_name FROM food_inspections LIMIT 1",
            conn
        )['dba_name'].iloc[0]
    
    # Calculate risk for the sample restaurant
    risk_data = calculator.calculate_risk_score(sample_restaurant)
    
    print(f"Restaurant: {sample_restaurant}")
    print(f"Risk Score: {risk_data['total_score']:.2f}")
    print(f"Risk Level: {calculator.get_risk_level(risk_data['total_score'])}")
    print("\nComponent Scores:")
    for component, score in risk_data['components'].items():
        print(f"{component}: {score:.2f}")

if __name__ == "__main__":
    main() 