import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

class CrashPredictionModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        
    def preprocess_data(self, df):
        """Preprocess the crash data"""
        # Select important columns based on your previous analysis
        important_cols = ['crash_date/time', 'acrs_report_type', 'driver_substance_abuse',
                         'speed_limit', 'vehicle_going_dir', 'driver_distracted_by']
        
        # Filter columns that exist in the dataframe
        available_cols = [col for col in important_cols if col in df.columns]
        df_filtered = df[available_cols].copy()
        
        # Drop rows with too many nulls
        df_filtered.dropna(thresh=3, inplace=True)
        
        # Fill missing values
        df_filtered.fillna('Unknown', inplace=True)
        
        # Handle datetime if present
        if 'crash_date/time' in df_filtered.columns:
            df_filtered['crash_date/time'] = pd.to_datetime(df_filtered['crash_date/time'], errors='coerce')
            df_filtered.dropna(subset=['crash_date/time'], inplace=True)
            df_filtered['crash_hour'] = df_filtered['crash_date/time'].dt.hour
            df_filtered['crash_day_of_week'] = df_filtered['crash_date/time'].dt.dayofweek
            df_filtered.drop('crash_date/time', axis=1, inplace=True)
        
        # Encode categorical features
        categorical_cols = df_filtered.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_filtered[col] = self.label_encoders[col].fit_transform(df_filtered[col])
            else:
                # Handle new categories during prediction
                try:
                    df_filtered[col] = self.label_encoders[col].transform(df_filtered[col])
                except ValueError:
                    # If new category, assign most frequent category
                    most_frequent = df_filtered[col].mode()[0] if not df_filtered[col].mode().empty else 'Unknown'
                    df_filtered[col] = df_filtered[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else most_frequent
                    )
                    df_filtered[col] = self.label_encoders[col].transform(df_filtered[col])
        
        return df_filtered
    
    def train_model(self, df):
        """Train the crash prediction model"""
        processed_df = self.preprocess_data(df)
        
        # Define target and features
        target_col = 'acrs_report_type' if 'acrs_report_type' in processed_df.columns else processed_df.columns[0]
        X = processed_df.drop([target_col], axis=1)
        y = processed_df[target_col]
        
        self.feature_columns = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest Model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure input_data has the same features as training data
        prediction_df = pd.DataFrame([input_data])
        
        # Add missing columns with default values
        for col in self.feature_columns:
            if col not in prediction_df.columns:
                prediction_df[col] = 0
        
        # Reorder columns to match training data
        prediction_df = prediction_df[self.feature_columns]
        
        prediction = self.model.predict(prediction_df)[0]
        probability = self.model.predict_proba(prediction_df)[0].max()
        
        return prediction, probability
    
    def save_model(self, filepath):
        """Save the trained model and encoders"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model and encoders"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            return True
        return False

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'crash_date/time': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'acrs_report_type': np.random.choice(['Property Damage', 'Injury', 'Fatal'], n_samples, p=[0.7, 0.25, 0.05]),
        'driver_substance_abuse': np.random.choice(['None', 'Alcohol', 'Drugs', 'Unknown'], n_samples, p=[0.8, 0.1, 0.05, 0.05]),
        'speed_limit': np.random.choice([25, 35, 45, 55, 65], n_samples),
        'vehicle_going_dir': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'driver_distracted_by': np.random.choice(['None', 'Cell Phone', 'Other', 'Unknown'], n_samples, p=[0.7, 0.15, 0.1, 0.05])
    }
    
    return pd.DataFrame(sample_data)