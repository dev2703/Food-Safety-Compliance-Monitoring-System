import sqlite3

def column_exists(conn, table, column):
    """Check if a column exists in a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]
    return column in columns

def update_schema():
    """Update database schema to include risk-related fields."""
    with sqlite3.connect('food_safety.db') as conn:
        # Add risk-related columns to restaurants table
        if not column_exists(conn, 'restaurants', 'risk_score'):
            conn.execute("""
                ALTER TABLE restaurants 
                ADD COLUMN risk_score FLOAT DEFAULT 0.0;
            """)
        if not column_exists(conn, 'restaurants', 'risk_level'):
            conn.execute("""
                ALTER TABLE restaurants 
                ADD COLUMN risk_level VARCHAR(20) DEFAULT 'Low';
            """)
        if not column_exists(conn, 'restaurants', 'last_risk_update'):
            conn.execute("""
                ALTER TABLE restaurants 
                ADD COLUMN last_risk_update TIMESTAMP;
            """)
        # Add risk-related columns to violations table
        if not column_exists(conn, 'violations', 'risk_impact'):
            conn.execute("""
                ALTER TABLE violations 
                ADD COLUMN risk_impact FLOAT DEFAULT 0.0;
            """)
        # Add risk-related columns to food_inspections table if they don't exist
        if not column_exists(conn, 'food_inspections', 'risk_score'):
            conn.execute("""
                ALTER TABLE food_inspections 
                ADD COLUMN risk_score FLOAT DEFAULT 0.0;
            """)
        if not column_exists(conn, 'food_inspections', 'risk_level'):
            conn.execute("""
                ALTER TABLE food_inspections 
                ADD COLUMN risk_level VARCHAR(20) DEFAULT 'Low';
            """)
        if not column_exists(conn, 'food_inspections', 'last_risk_update'):
            conn.execute("""
                ALTER TABLE food_inspections 
                ADD COLUMN last_risk_update TIMESTAMP;
            """)
        
        # Create inspection_outcomes table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspection_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dba_name TEXT NOT NULL,
                inspection_date DATE NOT NULL,
                outcome_type TEXT NOT NULL,
                fine_amount DECIMAL(10,2),
                closure_duration INTEGER,
                inspector_feedback TEXT,
                corrective_actions TEXT,
                follow_up_required BOOLEAN,
                follow_up_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dba_name) REFERENCES restaurants(dba_name),
                FOREIGN KEY (inspection_date) REFERENCES food_inspections(inspection_date)
            );
        """)
        
        conn.commit()
        print("Database schema updated successfully!")

if __name__ == "__main__":
    update_schema() 