# engine/ml_predictor.py
"""Machine Learning predictor for SOL trading"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.config import config
from utilities.logger import logger, log_prediction

class MLPredictor:
    """Machine Learning predictor for SOL price movements"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=4),
            'lr': LogisticRegression(random_state=42, max_iter=500, n_jobs=-1)
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.feature_importance = {}
        self.last_training = None
        self.performance_history = []
        
    def train_models(self, X, y):
        """Train all ML models"""
        if X is None or y is None or len(X) < 50:
            logger.warning('Insufficient data for training')
            return False
        
        try:
            logger.info(f'Training models with {len(X)} samples...')
            
            # Limit data size for faster training
            max_samples = 2000
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X = X[indices]
                y = y[indices]
                logger.info(f'Using {max_samples} samples for training')
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train each model
            for name, model in self.models.items():
                logger.info(f'Training {name} model...')
                
                # Train model
                model.fit(X_train_scaled, y_train)
                self.trained_models[name] = model
                
                # Test score only (skip CV for speed)
                test_score = model.score(X_test_scaled, y_test)
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                logger.info(f'Model {name}: Test Score: {test_score:.3f}')
            
            # Ensemble predictions on test set
            ensemble_pred = self._get_ensemble_prediction(X_test_scaled)
            if ensemble_pred[0] is not None:
                ensemble_accuracy = np.mean(ensemble_pred[0] == y_test)
                logger.info(f'Ensemble accuracy: {ensemble_accuracy:.3f}')
            
            # Store performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'accuracy': ensemble_accuracy if ensemble_pred[0] is not None else 0.5,
                'data_points': len(X_train)
            })
            
            self.last_training = datetime.now()
            self._save_models()
            
            return True
            
        except Exception as e:
            logger.error(f'Model training failed: {e}')
            return False
    
    def _get_ensemble_prediction(self, X):
        """Get ensemble prediction from all models"""
        if not self.trained_models:
            return None
        
        predictions = []
        confidences = []
        
        for name, model in self.trained_models.items():
            pred = model.predict(X)
            
            # Get prediction probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                conf = np.max(proba, axis=1)
            else:
                conf = np.ones(len(pred)) * 0.5
            
            predictions.append(pred)
            confidences.append(conf)
        
        # Weighted ensemble (weight by confidence)
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Ensemble prediction: majority vote with confidence weighting
        ensemble_pred = []
        ensemble_conf = []
        
        for i in range(predictions.shape[1]):
            # Get predictions and confidences for this sample
            sample_preds = predictions[:, i]
            sample_confs = confidences[:, i]
            
            # Weighted vote
            weighted_sum = np.sum(sample_preds * sample_confs)
            total_weight = np.sum(sample_confs)
            
            final_pred = 1 if weighted_sum / total_weight > 0.5 else 0
            final_conf = total_weight / len(sample_preds)
            
            ensemble_pred.append(final_pred)
            ensemble_conf.append(final_conf)
        
        return np.array(ensemble_pred), np.array(ensemble_conf)
    
    def predict(self, features):
        """Make prediction on new features"""
        if not self.trained_models or features is None:
            return 'HOLD', 0.0
        
        try:
            # Reshape if single sample
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get ensemble prediction
            pred, conf = self._get_ensemble_prediction(features_scaled)
            
            if pred is not None and len(pred) > 0:
                signal = 'BUY' if pred[0] == 1 else 'SELL'
                confidence = conf[0] if len(conf) > 0 else 0.5
                
                # Create features summary for logging
                try:
                    feature_summary = f'RSI:{features[0][0]:.1f}, MACD:{features[0][1]:.3f}, Vol:{features[0][4]:.2f}'
                except:
                    feature_summary = 'Features processed'
                
                # Only log predictions above threshold to reduce noise
                if confidence > config.PREDICTION_THRESHOLD:
                    log_prediction(signal, confidence, feature_summary)
                
                return signal, confidence
            
        except Exception as e:
            logger.error(f'Prediction failed: {e}')
        
        return 'HOLD', 0.0
    
    def should_retrain(self):
        """Check if models should be retrained"""
        if self.last_training is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_training).total_seconds() / 3600
        return hours_since_training >= config.MODEL_RETRAIN_HOURS
    
    def get_model_performance(self):
        """Get model performance metrics"""
        if not self.performance_history:
            return None
        
        recent_performance = self.performance_history[-10:]  # Last 10 trainings
        avg_accuracy = np.mean([p['accuracy'] for p in recent_performance])
        
        return {
            'average_accuracy': avg_accuracy,
            'last_training': self.last_training,
            'training_count': len(self.performance_history),
            'feature_importance': self.feature_importance
        }
    
    def _save_models(self):
        """Save trained models"""
        try:
            os.makedirs('models', exist_ok=True)
            
            model_data = {
                'trained_models': self.trained_models,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'last_training': self.last_training,
                'performance_history': self.performance_history
            }
            
            with open('models/sol_ml_models.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info('Models saved successfully')
            
        except Exception as e:
            logger.error(f'Failed to save models: {e}')
    
    def load_models(self):
        """Load trained models"""
        try:
            if os.path.exists('models/sol_ml_models.pkl'):
                with open('models/sol_ml_models.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.trained_models = model_data.get('trained_models', {})
                self.scaler = model_data.get('scaler', StandardScaler())
                self.feature_importance = model_data.get('feature_importance', {})
                self.last_training = model_data.get('last_training')
                self.performance_history = model_data.get('performance_history', [])
                
                logger.info(f'Loaded {len(self.trained_models)} trained models')
                return True
            
        except Exception as e:
            logger.error(f'Failed to load models: {e}')
        
        return False
    
    def analyze_feature_importance(self):
        """Analyze and log feature importance"""
        if not self.feature_importance:
            return
        
        feature_names = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'volume_ratio',
            'price_change_1h', 'price_change_4h', 'price_change_24h',
            'volatility', 'price_position', 'price_sma20_ratio',
            'price_sma50_ratio', 'sma_ratio'
        ]
        
        # Average importance across models
        if 'rf' in self.feature_importance:
            importance = self.feature_importance['rf']
            
            # Get top 5 most important features
            if len(importance) >= len(feature_names):
                feature_imp = list(zip(feature_names, importance))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                
                top_features = feature_imp[:5]
                logger.info(f'Top features: {", ".join([f"{name}:{imp:.3f}" for name, imp in top_features])}')

# Global instance
ml_predictor = MLPredictor()