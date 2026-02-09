"""
Predictive modeling module for energy demand forecasting
Demonstrates machine learning and business value
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Tuple, List


class EnergyForecaster:
    """
    Machine learning model to forecast energy consumption based on weather
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.metrics = {}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Engineer features for ML model"""
        features_df = df.copy()
        
        # Temperature features
        features_df['temp_squared'] = features_df['avg_temp'] ** 2
        features_df['temp_range_ratio'] = features_df['temp_range'] / (features_df['avg_temp'] + 1)
        
        # Time-based features
        features_df['day_of_week_num'] = features_df['date'].dt.dayofweek
        features_df['month_num'] = features_df['date'].dt.month
        features_df['day_of_year'] = features_df['date'].dt.dayofyear
        
        # Cyclical encoding for time features
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month_num'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month_num'] / 12)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week_num'] / 7)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week_num'] / 7)
        
        # City encoding (one-hot)
        city_dummies = pd.get_dummies(features_df['city'], prefix='city')
        features_df = pd.concat([features_df, city_dummies], axis=1)
        
        # Select features for model
        feature_cols = [
            'avg_temp', 'TMAX', 'TMIN', 'temp_range',
            'temp_squared', 'temp_range_ratio',
            'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos'
        ] + [col for col in features_df.columns if col.startswith('city_')]
        
        X = features_df[feature_cols]
        y = features_df['energy_consumption']
        
        return X, y
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train forecasting model and return performance metrics"""
        self.logger.info("Training energy forecasting model")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting model (better for this use case)
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        self.metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mape': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100,
            'test_mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='r2'
        )
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()
        
        self.logger.info(f"Model trained. Test R²: {self.metrics['test_r2']:.4f}, Test MAPE: {self.metrics['test_mape']:.2f}%")
        
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        X, _ = self.prepare_features(df)
        
        # Ensure all city columns from training are present
        # Get expected columns from the scaler
        expected_features = self.model.n_features_in_
        
        # Add missing city columns with value 0
        for city in ['New York', 'Chicago', 'Houston', 'Phoenix', 'Seattle']:
            col_name = f'city_{city}'
            if col_name not in X.columns:
                X[col_name] = 0
        
        # Ensure columns are in the same order as training
        X = X[self.scaler.feature_names_in_]
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def forecast_next_day(self, df: pd.DataFrame, forecast_temp: float, city: str, is_weekend: bool = False) -> Dict:
        """
        Forecast energy consumption for next day given temperature
        
        Returns: Dictionary with prediction and confidence interval
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get latest date
        latest_date = df['date'].max() + pd.Timedelta(days=1)
        
        # Create forecast row
        forecast_row = pd.DataFrame({
            'date': [latest_date],
            'avg_temp': [forecast_temp],
            'TMAX': [forecast_temp + 2],
            'TMIN': [forecast_temp - 2],
            'temp_range': [4],
            'city': [city],
            'is_weekend': [is_weekend],
            'energy_consumption': [0]  # placeholder
        })
        
        # Make prediction
        prediction = self.predict(forecast_row)[0]
        
        # Calculate confidence interval (using test RMSE as uncertainty)
        uncertainty = self.metrics.get('test_rmse', prediction * 0.1)
        
        return {
            'predicted_consumption': prediction,
            'lower_bound': prediction - 1.96 * uncertainty,
            'upper_bound': prediction + 1.96 * uncertainty,
            'forecast_date': latest_date.strftime('%Y-%m-%d'),
            'forecast_temp': forecast_temp,
            'city': city
        }
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N most important features"""
        if self.feature_importance is None:
            raise ValueError("Model not trained")
        
        return self.feature_importance.head(top_n)
    
    def calculate_business_impact(self, df: pd.DataFrame, cost_per_mwh: float = 50) -> Dict:
        """
        Calculate business impact of accurate forecasting
        
        Args:
            cost_per_mwh: Cost per megawatt-hour (default $50)
        """
        # Make predictions
        predictions = self.predict(df)
        actual = df['energy_consumption'].values
        
        # Calculate forecast errors
        errors = actual - predictions
        abs_errors = np.abs(errors)
        
        # Cost of forecast errors (over/under generation)
        # Overgeneration: wasted capacity
        # Undergeneration: emergency purchases at higher cost
        overgen_cost = np.sum(errors[errors > 0]) * cost_per_mwh * 0.3  # 30% cost for overgen
        undergen_cost = np.sum(-errors[errors < 0]) * cost_per_mwh * 2.0  # 200% cost for undergen
        
        total_error_cost = overgen_cost + undergen_cost
        
        # Potential savings with perfect forecast
        perfect_forecast_value = total_error_cost
        
        # Current model savings (vs naive baseline)
        naive_mae = np.mean(np.abs(actual - actual.mean()))
        model_mae = self.metrics['test_mae']
        improvement = (naive_mae - model_mae) / naive_mae
        estimated_savings = perfect_forecast_value * improvement
        
        return {
            'total_energy_mwh': actual.sum(),
            'total_error_cost': total_error_cost,
            'overgen_cost': overgen_cost,
            'undergen_cost': undergen_cost,
            'model_mae': model_mae,
            'naive_mae': naive_mae,
            'improvement_pct': improvement * 100,
            'estimated_annual_savings': estimated_savings * 365 / len(df),
            'cost_per_mwh': cost_per_mwh
        }


if __name__ == "__main__":
    # Test the forecaster
    import sys
    sys.path.append('..')
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_csv('../data/processed/combined_data.csv', parse_dates=['date'])
    
    # Train model
    forecaster = EnergyForecaster()
    metrics = forecaster.train_model(df)
    
    print("\n" + "="*70)
    print("ENERGY FORECASTING MODEL PERFORMANCE")
    print("="*70)
    print(f"Test R² Score: {metrics['test_r2']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:,.0f} MWh")
    print(f"Test MAPE: {metrics['test_mape']:.2f}%")
    print(f"Cross-Validation R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    
    print("\n" + "="*70)
    print("TOP 5 MOST IMPORTANT FEATURES")
    print("="*70)
    print(forecaster.get_feature_importance(5))
    
    # Business impact
    impact = forecaster.calculate_business_impact(df, cost_per_mwh=50)
    print("\n" + "="*70)
    print("BUSINESS IMPACT ANALYSIS")
    print("="*70)
    print(f"Model Improvement: {impact['improvement_pct']:.1f}% better than baseline")
    print(f"Estimated Annual Savings: ${impact['estimated_annual_savings']:,.0f}")
    
    # Example forecast
    forecast = forecaster.forecast_next_day(df, forecast_temp=45, city='New York')
    print("\n" + "="*70)
    print("SAMPLE FORECAST FOR NEW YORK (45°F)")
    print("="*70)
    print(f"Predicted: {forecast['predicted_consumption']:,.0f} MWh")
    print(f"95% CI: [{forecast['lower_bound']:,.0f}, {forecast['upper_bound']:,.0f}] MWh")
