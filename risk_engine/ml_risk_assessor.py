import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, metrics_file: str = 'risk_engine/models/performance_metrics.json'):
        self.metrics_file = metrics_file
        self.metrics_history = self._load_metrics()
        self.drift_threshold = 0.05  # 5% change in performance
        self.retrain_threshold = 0.1  # 10% degradation
        
    def _load_metrics(self) -> Dict:
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {'history': []}
    
    def _save_metrics(self):
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def update_metrics(self, metrics: Dict):
        """Update performance metrics and check for drift."""
        timestamp = datetime.now().isoformat()
        self.metrics_history['history'].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        self._save_metrics()
        
        # Check for drift
        if len(self.metrics_history['history']) > 1:
            return self._check_drift()
        return False
    
    def _check_drift(self) -> bool:
        """Check if model performance has drifted significantly."""
        recent_metrics = self.metrics_history['history'][-5:]  # Last 5 measurements
        if len(recent_metrics) < 2:
            return False
            
        # Calculate average performance
        avg_performance = np.mean([m['metrics']['auc_score'] for m in recent_metrics])
        latest_performance = recent_metrics[-1]['metrics']['auc_score']
        
        # Check for significant drift
        if (avg_performance - latest_performance) > self.drift_threshold:
            logger.warning(f"Model drift detected! Performance dropped by {avg_performance - latest_performance:.2%}")
            return True
        return False
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained based on performance degradation."""
        if len(self.metrics_history['history']) < 2:
            return False
            
        initial_performance = self.metrics_history['history'][0]['metrics']['auc_score']
        latest_performance = self.metrics_history['history'][-1]['metrics']['auc_score']
        
        if (initial_performance - latest_performance) > self.retrain_threshold:
            logger.warning(f"Model performance degraded by {(initial_performance - latest_performance):.2%}. Retraining recommended.")
            return True
        return False

class MLRiskAssessor:
    def __init__(self, db_path: str = 'food_safety.db'):
        """Initialize the ML-based risk assessor."""
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = IterativeImputer(
            estimator=xgb.XGBRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            max_iter=10,
            random_state=42,
            n_nearest_features=5,
            sample_posterior=True
        )
        self.label_encoders = {}
        self.feature_columns = []
        self.monitor = ModelMonitor()
        self.location_clusters = None
        self.violation_patterns = None
        
    def _load_data(self) -> pd.DataFrame:
        """Load and join all relevant data from the database."""
        with sqlite3.connect(self.db_path) as conn:
            # Load inspections with violations
            inspections = pd.read_sql("""
                SELECT 
                    i.*,
                    r."Type" as facility_type,
                    r."Address" as address,
                    r."City" as city,
                    r."State" as state,
                    r."Zip" as zip
                FROM food_inspections i
                LEFT JOIN restaurants r ON i."DBA Name" = r."Name"
            """, conn)
            
            # Load historical outcomes (closures, fines)
            outcomes = pd.read_sql("""
                SELECT 
                    dba_name,
                    inspection_date,
                    outcome_type,
                    fine_amount
                FROM inspection_outcomes
            """, conn)
            
            return inspections, outcomes

    def _extract_location_features(self, inspections: pd.DataFrame) -> pd.DataFrame:
        """Extract location-based features using clustering."""
        if self.location_clusters is None:
            # Extract coordinates and cluster locations
            coords = inspections[['Latitude', 'Longitude']].dropna()
            if not coords.empty:
                self.location_clusters = DBSCAN(eps=0.01, min_samples=5).fit(coords)
        
        if self.location_clusters is not None:
            inspections['location_cluster'] = self.location_clusters.labels_
            # Calculate cluster statistics
            cluster_stats = inspections.groupby('location_cluster').agg({
                'total_violations': ['mean', 'std', 'count'],
                'critical_count': 'mean'
            }).fillna(0)
            
            # Add cluster-based features
            inspections = inspections.merge(
                cluster_stats,
                on='location_cluster',
                how='left',
                suffixes=('', '_cluster')
            )
        
        return inspections

    def _extract_violation_patterns(self, inspections: pd.DataFrame) -> pd.DataFrame:
        """Extract violation co-occurrence patterns."""
        if self.violation_patterns is None:
            # Extract unique violation types
            all_violations = set()
            for violations in inspections['Violations'].dropna():
                all_violations.update(violations.split('|'))
            
            # Create violation co-occurrence matrix
            violation_matrix = pd.DataFrame(0, 
                index=inspections.index,
                columns=list(all_violations)
            )
            
            for idx, violations in inspections['Violations'].dropna().items():
                for violation in violations.split('|'):
                    violation_matrix.loc[idx, violation] = 1
            
            # Use PCA to reduce dimensionality of violation patterns
            pca = PCA(n_components=5)
            self.violation_patterns = pca.fit_transform(violation_matrix)
            
            # Add violation pattern features
            for i in range(5):
                inspections[f'violation_pattern_{i+1}'] = self.violation_patterns[:, i]
        
        return inspections

    def _extract_seasonal_features(self, inspections: pd.DataFrame) -> pd.DataFrame:
        """Extract seasonal and temporal patterns."""
        # Add seasonal indicators
        inspections['season'] = inspections['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Add holiday indicators
        holidays = {
            1: ['New Year', 'MLK Day'],
            2: ['Presidents Day'],
            3: ['St Patricks Day'],
            4: ['Easter'],
            5: ['Memorial Day'],
            7: ['Independence Day'],
            9: ['Labor Day'],
            10: ['Columbus Day'],
            11: ['Veterans Day', 'Thanksgiving'],
            12: ['Christmas']
        }
        
        inspections['is_holiday'] = inspections['month'].map(
            lambda x: 1 if x in holidays else 0
        )
        
        # Add time-based features
        inspections['time_since_last_inspection'] = inspections.groupby('DBA Name')['Inspection Date'].diff().dt.days
        inspections['inspection_frequency'] = inspections.groupby('DBA Name')['Inspection Date'].transform(
            lambda x: 365 / (x.max() - x.min()).days if (x.max() - x.min()).days > 0 else 0
        )
        
        # Add rolling statistics
        for window in [3, 6, 12]:
            inspections[f'rolling_violations_{window}m'] = inspections.groupby('DBA Name')['total_violations'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            inspections[f'rolling_critical_{window}m'] = inspections.groupby('DBA Name')['critical_count'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        return inspections

    def _extract_establishment_features(self, inspections: pd.DataFrame) -> pd.DataFrame:
        """Extract establishment-specific patterns."""
        # Calculate establishment-specific statistics
        establishment_stats = inspections.groupby('DBA Name').agg({
            'total_violations': ['mean', 'std', 'max'],
            'critical_count': ['mean', 'max'],
            'Inspection Date': ['min', 'max', 'count']
        }).fillna(0)
        
        # Add establishment-based features
        inspections = inspections.merge(
            establishment_stats,
            on='DBA Name',
            how='left',
            suffixes=('', '_establishment')
        )
        
        # Calculate risk trends
        inspections['violation_trend'] = inspections.groupby('DBA Name')['total_violations'].transform(
            lambda x: x.diff().rolling(window=3, min_periods=1).mean()
        )
        
        # Calculate risk volatility
        inspections['risk_volatility'] = inspections.groupby('DBA Name')['total_violations'].transform(
            lambda x: x.rolling(window=6, min_periods=1).std()
        )
        
        return inspections

    def _extract_features(self, inspections: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer features from the raw data."""
        # Convert dates
        inspections['Inspection Date'] = pd.to_datetime(inspections['Inspection Date'])
        if not outcomes.empty:
            outcomes['inspection_date'] = pd.to_datetime(outcomes['inspection_date'])
        
        # Extract base features
        inspections = self._extract_location_features(inspections)
        inspections = self._extract_violation_patterns(inspections)
        inspections = self._extract_seasonal_features(inspections)
        inspections = self._extract_establishment_features(inspections)
        
        # Calculate violation patterns
        def extract_violation_patterns(violations_str):
            if pd.isna(violations_str):
                return {
                    'critical_count': 0,
                    'serious_count': 0,
                    'minor_count': 0,
                    'total_violations': 0,
                    'critical_ratio': 0,
                    'serious_ratio': 0,
                    'minor_ratio': 0,
                    'violation_diversity': 0,
                    'violation_severity': 0
                }
            
            violations = violations_str.split('|')
            total = len(violations)
            critical = sum(1 for v in violations if 'CRITICAL' in v.upper())
            serious = sum(1 for v in violations if 'SERIOUS' in v.upper())
            minor = sum(1 for v in violations if 'MINOR' in v.upper())
            
            # Calculate violation diversity and severity
            unique_violations = len(set(violations))
            violation_diversity = unique_violations / total if total > 0 else 0
            violation_severity = (critical * 3 + serious * 2 + minor) / total if total > 0 else 0
            
            return {
                'critical_count': critical,
                'serious_count': serious,
                'minor_count': minor,
                'total_violations': total,
                'critical_ratio': critical/total if total > 0 else 0,
                'serious_ratio': serious/total if total > 0 else 0,
                'minor_ratio': minor/total if total > 0 else 0,
                'violation_diversity': violation_diversity,
                'violation_severity': violation_severity
            }
        
        violation_patterns = inspections['Violations'].apply(extract_violation_patterns)
        violation_df = pd.DataFrame(violation_patterns.tolist())
        inspections = pd.concat([inspections, violation_df], axis=1)
        
        # Merge with outcomes if not empty
        if not outcomes.empty:
            inspections = pd.merge(
                inspections,
                outcomes,
                left_on=['DBA Name', 'Inspection Date'],
                right_on=['dba_name', 'inspection_date'],
                how='left'
            )
        
        # Encode categorical variables
        categorical_columns = ['facility_type', 'city', 'state', 'outcome_type', 'season']
        for col in categorical_columns:
            if col in inspections.columns:
                self.label_encoders[col] = LabelEncoder()
                inspections[col] = self.label_encoders[col].fit_transform(
                    inspections[col].fillna('Unknown')
                )
        
        # Select features for model
        feature_columns = [
            'facility_type', 'city', 'state', 'month', 'day_of_week', 'quarter', 'year',
            'is_weekend', 'is_holiday', 'season',
            'critical_count', 'serious_count', 'minor_count', 'total_violations',
            'critical_ratio', 'serious_ratio', 'minor_ratio', 'violation_diversity',
            'violation_severity', 'days_since_last', 'inspection_frequency',
            'rolling_critical_avg', 'rolling_violations_avg', 'rolling_violation_diversity',
            'violation_trend', 'risk_volatility'
        ]
        
        # Add location cluster features
        if 'location_cluster' in inspections.columns:
            feature_columns.extend([
                'total_violations_mean_cluster', 'total_violations_std_cluster',
                'total_violations_count_cluster', 'critical_count_mean_cluster'
            ])
        
        # Add violation pattern features
        feature_columns.extend([f'violation_pattern_{i+1}' for i in range(5)])
        
        # Add establishment features
        feature_columns.extend([
            'total_violations_mean_establishment', 'total_violations_std_establishment',
            'total_violations_max_establishment', 'critical_count_mean_establishment',
            'critical_count_max_establishment'
        ])
        
        # Add rolling window features
        for window in [3, 6, 12]:
            feature_columns.extend([
                f'rolling_violations_{window}m',
                f'rolling_critical_{window}m'
            ])
        
        self.feature_columns = [col for col in feature_columns if col in inspections.columns]
        
        return inspections[self.feature_columns], inspections.get('outcome_type', pd.Series([None]*len(inspections)))

    def train(self, test_size: float = 0.2, random_state: int = 42):
        """Train the stacked ensemble model."""
        logger.info("Loading and preprocessing data...")
        inspections, outcomes = self._load_data()
        X, y = self._extract_features(inspections, outcomes)
        
        # Impute missing values using IterativeImputer
        logger.info("Imputing missing values using IterativeImputer...")
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define base models
        base_models = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=random_state
            ))
        ]
        
        # Create stacking classifier
        logger.info("Training stacked ensemble model...")
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Update monitoring metrics
        self.monitor.update_metrics(metrics)
        
        logger.info("\nModel Performance:")
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"AUC Score: {metrics['auc_score']:.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        logger.info(f"\nCross-validation scores: {cv_scores}")
        logger.info(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Save model and scaler
        joblib.dump(self.model, 'risk_engine/models/risk_model.joblib')
        joblib.dump(self.scaler, 'risk_engine/models/scaler.joblib')
        joblib.dump(self.imputer, 'risk_engine/models/imputer.joblib')
        joblib.dump(self.label_encoders, 'risk_engine/models/label_encoders.joblib')
        
        return self.model

    def predict_risk(self, dba_name: str) -> Dict:
        """Predict risk for a specific establishment."""
        if self.model is None:
            self.load_model()
        
        # Get latest inspection data
        with sqlite3.connect(self.db_path) as conn:
            latest_inspection = pd.read_sql("""
                SELECT 
                    i.*,
                    r."Type" as facility_type,
                    r."Address" as address,
                    r."City" as city,
                    r."State" as state,
                    r."Zip" as zip
                FROM food_inspections i
                LEFT JOIN restaurants r ON i."DBA Name" = r."Name"
                WHERE i."DBA Name" = ?
                ORDER BY i."Inspection Date" DESC
                LIMIT 1
            """, conn, params=(dba_name,))
        
        if latest_inspection.empty:
            return {
                'risk_score': 0.0,
                'risk_level': 'Unknown',
                'confidence': 0.0,
                'factors': []
            }
        
        # Prepare features
        X, _ = self._extract_features(latest_inspection, pd.DataFrame())
        
        # Impute missing values
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probabilities
        risk_level = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)
        
        # Get feature importance from the best base model (XGBoost)
        xgb_model = self.model.named_estimators_['xgb']
        feature_importance = dict(zip(
            self.feature_columns,
            xgb_model.feature_importances_
        ))
        
        # Convert risk level back to original label
        risk_level = self.label_encoders['outcome_type'].inverse_transform([risk_level])[0]
        
        return {
            'risk_score': float(confidence),
            'risk_level': risk_level,
            'confidence': float(confidence),
            'factors': [
                {'feature': k, 'importance': float(v)}
                for k, v in sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ]
        }

    def load_model(self):
        """Load the trained model and associated objects."""
        try:
            self.model = joblib.load('risk_engine/models/risk_model.joblib')
            self.scaler = joblib.load('risk_engine/models/scaler.joblib')
            self.imputer = joblib.load('risk_engine/models/imputer.joblib')
            self.label_encoders = joblib.load('risk_engine/models/label_encoders.joblib')
        except FileNotFoundError:
            logger.error("Model files not found. Please train the model first.")
            raise

    def check_model_health(self) -> Dict:
        """Check model health and performance metrics."""
        return {
            'metrics_history': self.monitor.metrics_history,
            'needs_retraining': self.monitor.should_retrain(),
            'has_drift': self.monitor._check_drift()
        }

def main():
    """Example usage of the ML risk assessor."""
    # Create models directory if it doesn't exist
    import os
    os.makedirs('risk_engine/models', exist_ok=True)
    
    # Train model
    assessor = MLRiskAssessor()
    assessor.train()
    
    # Check model health
    health_status = assessor.check_model_health()
    logger.info("\nModel Health Status:")
    logger.info(f"Needs Retraining: {health_status['needs_retraining']}")
    logger.info(f"Has Drift: {health_status['has_drift']}")
    
    # Test prediction
    with sqlite3.connect('food_safety.db') as conn:
        sample_restaurant = pd.read_sql(
            "SELECT DISTINCT \"DBA Name\" FROM food_inspections LIMIT 1",
            conn
        )["DBA Name"].iloc[0]
    
    risk_assessment = assessor.predict_risk(sample_restaurant)
    
    print(f"\nRisk Assessment for {sample_restaurant}:")
    print(f"Risk Level: {risk_assessment['risk_level']}")
    print(f"Confidence: {risk_assessment['confidence']:.2f}")
    print("\nTop Risk Factors:")
    for factor in risk_assessment['factors']:
        print(f"- {factor['feature']}: {factor['importance']:.3f}")

if __name__ == "__main__":
    main() 