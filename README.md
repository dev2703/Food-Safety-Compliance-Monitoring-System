# Food Safety Compliance Monitoring System (Sentinel)

A full-stack system for monitoring and predicting food safety compliance risks across food delivery operations.

## Project Overview

Sentinel combines real inspection data with simulated audit logs to predict non-compliance risks in food delivery operations. The system uses machine learning to identify patterns and potential violations before they occur.

### Key Features

- Real-time compliance risk prediction
- Historical inspection data analysis
- Synthetic data generation for testing
- Interactive dashboard for monitoring
- Automated audit logging
- Risk heatmaps and trend analysis

## Tech Stack

- **Backend**: Python, dbt
- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn, XGBoost
- **Database**: DuckDB/PostgreSQL
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, Seaborn

## Project Structure

```
├── data/
│   ├── raw/           # Original datasets
│   ├── processed/     # Cleaned data
│   └── simulated/     # Generated synthetic data
├── data_engineering/
│   ├── ingest.py      # Data loading utilities
│   └── simulate.py    # Synthetic data generation
├── sentinel_dbt/      # dbt transformation pipeline
│   ├── models/staging/
│   ├── models/intermediate/
│   └── models/marts/
├── ml/
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate.py
├── dashboard/
│   ├── app.py         # Main Streamlit app
│   └── pages/         # Multi-page components
├── notebooks/         # Jupyter analysis notebooks
├── tests/            # Unit and integration tests
└── models/           # Saved ML artifacts
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/dev2703/Food-Safety-Compliance-Monitoring-System.git
cd Food-Safety-Compliance-Monitoring-System
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
# Instructions for database setup will be added
```

5. Run the development server:
```bash
streamlit run dashboard/app.py
```

## Development Milestones

1. 🏗️ Project Setup & Data Acquisition
2. 🎭 Data Simulation & Ingestion
3. 🧹 Data Cleaning & Transformation
4. ⚙️ Feature Engineering & ML Pipeline
5. 🤖 Model Training & Evaluation
6. 📊 Dashboard Development & Deployment

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or concerns, please open an issue in the GitHub repository. 