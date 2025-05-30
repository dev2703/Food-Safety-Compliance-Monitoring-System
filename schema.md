# Data Schema Documentation

## Overview

This document outlines the data schema for the Food Safety Compliance Monitoring System. The system integrates real inspection data with simulated audit logs to provide comprehensive compliance monitoring.

## Data Sources

### Real Inspection Data
- NYC Food Inspection Data
- USDA Food Safety Data
- Chicago Food Inspection Data

### Simulated Data
- Store Profiles
- Audit Logs
- Inspector Data
- Environmental Factors

## Schema Definitions

### 1. Store Profiles
```sql
CREATE TABLE store_profiles (
    store_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    address TEXT,
    city VARCHAR(50),
    state VARCHAR(2),
    zip_code VARCHAR(10),
    cuisine_type VARCHAR(50),
    establishment_type VARCHAR(50),
    seating_capacity INTEGER,
    opening_date DATE,
    last_inspection_date DATE,
    risk_level VARCHAR(20)
);
```

### 2. Inspection Records
```sql
CREATE TABLE inspection_records (
    inspection_id VARCHAR(50) PRIMARY KEY,
    store_id VARCHAR(50) REFERENCES store_profiles(store_id),
    inspection_date DATE,
    inspector_id VARCHAR(50),
    score INTEGER,
    grade VARCHAR(2),
    violation_count INTEGER,
    critical_violations INTEGER,
    non_critical_violations INTEGER
);
```

### 3. Violations
```sql
CREATE TABLE violations (
    violation_id VARCHAR(50) PRIMARY KEY,
    inspection_id VARCHAR(50) REFERENCES inspection_records(inspection_id),
    violation_code VARCHAR(20),
    violation_description TEXT,
    critical_flag BOOLEAN,
    violation_category VARCHAR(50)
);
```

### 4. Audit Logs
```sql
CREATE TABLE audit_logs (
    audit_id VARCHAR(50) PRIMARY KEY,
    store_id VARCHAR(50) REFERENCES store_profiles(store_id),
    auditor_id VARCHAR(50),
    audit_date TIMESTAMP,
    checklist_id VARCHAR(50),
    completion_status VARCHAR(20),
    total_items INTEGER,
    passed_items INTEGER,
    failed_items INTEGER
);
```

### 5. Environmental Factors
```sql
CREATE TABLE environmental_factors (
    record_id VARCHAR(50) PRIMARY KEY,
    store_id VARCHAR(50) REFERENCES store_profiles(store_id),
    timestamp TIMESTAMP,
    temperature FLOAT,
    humidity FLOAT,
    weather_condition VARCHAR(50),
    power_outage BOOLEAN,
    equipment_failure BOOLEAN
);
```

## Data Relationships

1. **Store Profiles → Inspection Records**: One-to-Many
   - Each store can have multiple inspection records
   - Each inspection record belongs to one store

2. **Inspection Records → Violations**: One-to-Many
   - Each inspection can have multiple violations
   - Each violation belongs to one inspection

3. **Store Profiles → Audit Logs**: One-to-Many
   - Each store can have multiple audit logs
   - Each audit log belongs to one store

4. **Store Profiles → Environmental Factors**: One-to-Many
   - Each store can have multiple environmental factor records
   - Each environmental factor record belongs to one store

## Data Quality Rules

1. **Store Profiles**
   - `store_id` must be unique
   - `name` cannot be null
   - `address` must be complete
   - `risk_level` must be one of: 'Low', 'Medium', 'High'

2. **Inspection Records**
   - `score` must be between 0 and 100
   - `grade` must be one of: 'A', 'B', 'C', 'D', 'F'
   - `violation_count` must equal sum of critical and non-critical violations

3. **Violations**
   - `violation_code` must follow standard format
   - `critical_flag` must be boolean
   - `violation_category` must be from predefined list

4. **Audit Logs**
   - `completion_status` must be one of: 'Complete', 'Incomplete', 'Failed'
   - `total_items` must equal sum of passed and failed items

5. **Environmental Factors**
   - `temperature` must be in Celsius
   - `humidity` must be between 0 and 100
   - `timestamp` must be in UTC

## Data Dictionary

### Key Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| store_id | VARCHAR(50) | Unique identifier for each store | "STORE_001" |
| inspection_id | VARCHAR(50) | Unique identifier for each inspection | "INSP_2023_001" |
| violation_code | VARCHAR(20) | Standardized code for violations | "4A" |
| audit_id | VARCHAR(50) | Unique identifier for each audit | "AUDIT_2023_001" |
| risk_level | VARCHAR(20) | Current risk assessment level | "High" |

## Data Flow

1. Raw data ingestion from multiple sources
2. Data cleaning and standardization
3. Feature engineering and transformation
4. Model training and prediction
5. Dashboard visualization and reporting

## Future Considerations

1. Add support for multiple languages
2. Implement real-time data streaming
3. Add support for image/video evidence
4. Implement geospatial data
5. Add support for mobile app integration 