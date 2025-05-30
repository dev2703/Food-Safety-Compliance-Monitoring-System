# Food Safety Risk Assessment System

A machine learning-based system for assessing food safety risks in restaurants and food establishments. The system uses advanced ML techniques to predict potential violations and assess risk levels based on historical inspection data.

## Features

- **Advanced Risk Assessment:**
  - Stacked ensemble of XGBoost, LightGBM, Neural Network, and Random Forest models
  - Sophisticated feature engineering including location-based, temporal, and pattern analysis
  - Real-time risk scoring and prediction

- **Sophisticated Feature Engineering:**
  - Location-based clustering and analysis
  - Violation pattern analysis using PCA
  - Seasonal and temporal pattern detection
  - Establishment-specific risk indicators
  - Historical performance tracking

- **Model Monitoring and Maintenance:**
  - Continuous performance tracking
  - Drift detection
  - Automated retraining triggers
  - Model health monitoring
  - Performance metrics logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/food-safety-risk-assessment.git
cd food-safety-risk-assessment
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

## Usage

1. Train the model:
```bash
python risk_engine/ml_risk_assessor.py
```

2. Check model health:
```python
from risk_engine.ml_risk_assessor import MLRiskAssessor

assessor = MLRiskAssessor()
health_status = assessor.check_model_health()
print(health_status)
```

3. Predict risk for an establishment:
```python
risk_assessment = assessor.predict_risk("Restaurant Name")
print(risk_assessment)
```

## Project Structure

```
food-safety-risk-assessment/
├── risk_engine/
│   ├── ml_risk_assessor.py      # Main ML risk assessment module
│   ├── risk_calculator.py       # Risk calculation utilities
│   ├── update_schema.py         # Database schema management
│   └── models/                  # Saved model files
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .gitignore                  # Git ignore file
```

## Dependencies

- Python 3.8+
- scikit-learn
- XGBoost
- LightGBM
- pandas
- numpy
- scipy
- joblib

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Food safety inspection data providers
- Open source machine learning community
- Contributors and maintainers 