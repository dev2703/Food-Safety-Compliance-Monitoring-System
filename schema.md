# Food Safety Compliance Monitoring System - Data Schema

## 1. Food Inspections Dataset
Primary dataset containing food inspection records.

| Column | Type | Description |
|--------|------|-------------|
| Inspection ID | Integer | Unique identifier for each inspection |
| DBA Name | String | Restaurant/establishment name |
| AKA Name | String | Alternative name for the establishment |
| License # | String | Business license number |
| Facility Type | String | Type of food establishment |
| Risk | String | Risk level (High/Medium/Low) |
| Address | String | Physical address |
| City | String | City name |
| State | String | State code |
| Zip | String | ZIP code |
| Inspection Date | Date | Date of inspection |
| Inspection Type | String | Type of inspection (Routine/Complaint/etc.) |
| Results | String | Inspection result (Pass/Fail/Pass w/ Conditions) |
| Violations | Text | Detailed violation descriptions |
| Latitude | Float | Geographic latitude |
| Longitude | Float | Geographic longitude |

## 2. Restaurants Dataset (Simulated)
Contains detailed information about food establishments.

| Column | Type | Description |
|--------|------|-------------|
| Restaurant ID | Integer | Unique identifier for each restaurant |
| Name | String | Restaurant name |
| Type | String | Type of establishment |
| Risk Level | String | Risk level (High/Medium/Low) |
| Address | String | Physical address |
| City | String | City name |
| State | String | State code |
| Zip | String | ZIP code |
| License Number | String | Business license number |
| Opening Date | Date | Date when restaurant opened |
| Capacity | Integer | Maximum seating capacity |
| Has Delivery | Boolean | Whether restaurant offers delivery |
| Has Outdoor Seating | Boolean | Whether restaurant has outdoor seating |

## 3. Violations Dataset (Simulated)
Records of specific violations found during inspections.

| Column | Type | Description |
|--------|------|-------------|
| Violation ID | Integer | Unique identifier for each violation |
| Restaurant ID | Integer | Reference to restaurant |
| Inspection ID | Integer | Reference to inspection |
| Violation Type | String | Category of violation |
| Severity | String | Severity level (Critical/Serious/Minor) |
| Description | Text | Detailed violation description |
| Date | Date | Date violation was recorded |
| Corrected | Boolean | Whether violation was corrected |
| Correction Date | Date | Date when violation was corrected |

## 4. Historical Inspections Dataset (Simulated)
Historical record of all inspections.

| Column | Type | Description |
|--------|------|-------------|
| Inspection ID | Integer | Unique identifier for each inspection |
| Restaurant ID | Integer | Reference to restaurant |
| Inspection Type | String | Type of inspection |
| Date | Date | Date of inspection |
| Result | String | Inspection result |
| Score | Integer | Numerical inspection score |
| Inspector ID | String | Identifier for inspector |
| Notes | Text | Additional inspection notes |

## Relationships
- Restaurants (1) → (Many) Inspections
- Inspections (1) → (Many) Violations
- Restaurants (1) → (Many) Historical Inspections

## Data Quality Notes
- All dates are in YYYY-MM-DD format
- Coordinates are in decimal degrees
- Risk levels are standardized across datasets
- Inspection results are categorized consistently 