import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import random

def generate_outcomes():
    """Generate sample inspection outcomes data."""
    with sqlite3.connect('food_safety.db') as conn:
        # Get existing inspections
        inspections = pd.read_sql("""
            SELECT "DBA Name" as dba_name, 
                   "Inspection Date" as inspection_date, 
                   Violations as violations
            FROM food_inspections
        """, conn)
        
        # Define possible outcomes and their probabilities
        outcomes = {
            'Pass': 0.6,
            'Pass with Conditions': 0.2,
            'Fail': 0.15,
            'Closure': 0.05
        }
        
        # Generate outcomes for each inspection
        inspection_outcomes = []
        for _, row in inspections.iterrows():
            # Determine outcome based on violations
            violations = row['violations']
            if pd.isna(violations):
                outcome = np.random.choice(
                    list(outcomes.keys()),
                    p=list(outcomes.values())
                )
            else:
                # If there are violations, increase probability of negative outcomes
                if 'CRITICAL' in violations.upper():
                    outcome = np.random.choice(
                        ['Fail', 'Closure'],
                        p=[0.7, 0.3]
                    )
                elif 'SERIOUS' in violations.upper():
                    outcome = np.random.choice(
                        ['Pass with Conditions', 'Fail'],
                        p=[0.6, 0.4]
                    )
                else:
                    outcome = np.random.choice(
                        ['Pass', 'Pass with Conditions'],
                        p=[0.8, 0.2]
                    )
            
            # Generate additional outcome details
            fine_amount = 0.0
            closure_duration = None
            follow_up_required = False
            follow_up_date = None
            
            if outcome == 'Fail':
                fine_amount = random.uniform(100, 1000)
                follow_up_required = True
                follow_up_date = pd.to_datetime(row['inspection_date']) + timedelta(days=random.randint(7, 30))
            elif outcome == 'Closure':
                fine_amount = random.uniform(1000, 5000)
                closure_duration = random.randint(1, 14)
                follow_up_required = True
                follow_up_date = pd.to_datetime(row['inspection_date']) + timedelta(days=closure_duration + 1)
            elif outcome == 'Pass with Conditions':
                follow_up_required = random.random() < 0.3
                if follow_up_required:
                    follow_up_date = pd.to_datetime(row['inspection_date']) + timedelta(days=random.randint(14, 60))
            
            # Generate inspector feedback
            feedback_templates = {
                'Pass': [
                    "Establishment maintained good standards.",
                    "No significant issues found.",
                    "Clean and well-maintained facility."
                ],
                'Pass with Conditions': [
                    "Minor issues need attention.",
                    "Some areas need improvement.",
                    "Basic standards met with room for improvement."
                ],
                'Fail': [
                    "Multiple violations found.",
                    "Significant improvements needed.",
                    "Failed to meet basic standards."
                ],
                'Closure': [
                    "Immediate closure required due to critical violations.",
                    "Serious health hazards present.",
                    "Facility unsafe for operation."
                ]
            }
            
            inspector_feedback = random.choice(feedback_templates[outcome])
            
            # Generate corrective actions
            corrective_actions = None
            if outcome in ['Fail', 'Closure', 'Pass with Conditions']:
                corrective_actions = "|".join([
                    "Update food safety procedures",
                    "Train staff on proper handling",
                    "Improve cleaning protocols",
                    "Maintain temperature logs",
                    "Fix equipment issues"
                ][:random.randint(1, 3)])
            
            inspection_outcomes.append({
                'dba_name': row['dba_name'],
                'inspection_date': row['inspection_date'],
                'outcome_type': outcome,
                'fine_amount': fine_amount,
                'closure_duration': closure_duration,
                'inspector_feedback': inspector_feedback,
                'corrective_actions': corrective_actions,
                'follow_up_required': follow_up_required,
                'follow_up_date': follow_up_date
            })
        
        # Convert to DataFrame and save to database
        outcomes_df = pd.DataFrame(inspection_outcomes)
        outcomes_df.to_sql('inspection_outcomes', conn, if_exists='replace', index=False)
        print(f"Generated {len(outcomes_df)} inspection outcomes")

if __name__ == "__main__":
    generate_outcomes() 