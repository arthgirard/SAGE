# engine/ml_predictor.py
"""Machine Learning predictor for SOL trading with persistent learning"""

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

class PersistentMLPredictor:
    """Machine Learning predictor with persistent learning across all runs"""
    
    def __init__(self):
        # Fixed model configuration for consistency
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10, 
                n_jobs=-1,
                max_features='sqrt',
                bootstrap=True
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=6,
                subsample=0.8,
                learning_rate=0.1
            ),
            'lr': LogisticRegression(
                random_state=42,
                max_iter=1000, 
                n_jobs=-1,
                C=1.0
            )
        }
        
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.feature_importance = {}
        self.last_training = None
        self.performance_history = []
        self.training_count = 0
        self.cumulative_training_data = {'X': [], 'y': []}
        
        # Performance tracking for model evolution
        self.model_versions = []
        self.best_performance = 0.0
        self.model_improvements = []
        
    def initialize(self):
        """Initialize by loading existing model or creating new one"""
        if self.load_models():
            logger.info(f'Loaded persistent model (training #{self.training_count})')
            logger.info(f'Best performance so far: {self.best_performance:.1%}')
        else:
            logger.info('No existing model found - will train new persistent model')
        
    def train_models(self, X, y):
        """Train models with cumulative learning approach"""
        if X is None or y is None or len(X) < 50:
            logger.warning('Insufficient data for training')
            return False
        
        try:
            logger.info(f'Training persistent model with {len(X)} new samples...')
            
            # Add new data to cumulative dataset
            if len(self.cumulative_training_data['X']) > 0:
                # Combine with existing data
                X_combined = np.vstack([self.cumulative_training_data['X'], X])
                y_combined = np.hstack([self.cumulative_training_data['y'], y])
                logger.info(f'Combined with {len(self.cumulative_training_data["X"])} existing samples')
            else:
                X_combined = X
                y_combined = y
            
            # Limit total dataset size to prevent memory issues
            max_total_samples = 10000
            if len(X_combined) > max_total_samples:
                # Keep most recent data
                keep_indices = np.arange(len(X_combined) - max_total_samples, len(X_combined))
                X_combined = X_combined[keep_indices]
                y_combined = y_combined[keep_indices]
                logger.info(f'Limited to {max_total_samples} most recent samples')
            
            # Update cumulative data
            self.cumulative_training_data['X'] = X_combined
            self.cumulative_training_data['y'] = y_combined
            
            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train each model
            model_scores = {}
            for name, model in self.models.items():
                logger.info(f'Training {name} model...')
                
                # Train model
                model.fit(X_train_scaled, y_train)
                self.trained_models[name] = model
                
                # Validation score
                val_score = model.score(X_test_scaled, y_test)
                model_scores[name] = val_score
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                logger.info(f'Model {name}: Validation Score: {val_score:.3f}')
            
            # Ensemble validation
            ensemble_pred = self._get_ensemble_prediction(X_test_scaled)
            if ensemble_pred[0] is not None:
                ensemble_accuracy = np.mean(ensemble_pred[0] == y_test)
                logger.info(f'Ensemble validation accuracy: {ensemble_accuracy:.3f}')
                
                # Track model improvement
                if ensemble_accuracy > self.best_performance:
                    improvement = ensemble_accuracy - self.best_performance
                    self.best_performance = ensemble_accuracy
                    self.model_improvements.append({
                        'training_id': self.training_count + 1,
                        'accuracy': ensemble_accuracy,
                        'improvement': improvement,
                        'timestamp': datetime.now(),
                        'total_samples': len(X_combined)
                    })
                    logger.info(f'NEW BEST MODEL! Accuracy improved by {improvement:.1%} to {ensemble_accuracy:.1%}')
                else:
                    logger.info(f'Model performance: {ensemble_accuracy:.1%} (best: {self.best_performance:.1%})')
            else:
                ensemble_accuracy = 0.5
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'accuracy': ensemble_accuracy,
                'data_points': len(X_combined),
                'model_scores': model_scores,
                'training_id': self.training_count + 1,
                'is_best': ensemble_accuracy >= self.best_performance
            })
            
            self.training_count += 1
            self.last_training = datetime.now()
            
            # Save the improved model
            self._save_models()
            
            return True
            
        except Exception as e:
            logger.error(f'Model training failed: {e}')
            return False
    
    def _get_ensemble_prediction(self, X):
        """Get ensemble prediction from all models"""
        if not self.trained_models:
            return None, None
        
        predictions = []
        confidences = []
        
        for name, model in self.trained_models.items():
            pred = model.predict(X)
            
            # Get prediction probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                conf = np.max(proba, axis=1)
            else:
                conf = np.ones(len(pred)) * 0.6
            
            predictions.append(pred)
            confidences.append(conf)
        
        # Ensemble prediction with equal weighting
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        ensemble_pred = []
        ensemble_conf = []
        
        for i in range(predictions.shape[1]):
            sample_preds = predictions[:, i]
            sample_confs = confidences[:, i]
            
            # Majority vote
            final_pred = 1 if np.mean(sample_preds) > 0.5 else 0
            final_conf = np.mean(sample_confs)
            
            ensemble_pred.append(final_pred)
            ensemble_conf.append(final_conf)
        
        return np.array(ensemble_pred), np.array(ensemble_conf)
    
    def predict(self, features):
        """Make prediction using the persistent model"""
        if not self.trained_models or features is None:
            return 'HOLD', 0.5
        
        try:
            # Reshape if single sample
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get ensemble prediction
            pred, conf = self._get_ensemble_prediction(features_scaled)
            
            if pred is not None and len(pred) > 0:
                confidence = conf[0] if len(conf) > 0 else 0.5
                prediction = pred[0]
                
                # Use configured threshold
                threshold = config.PREDICTION_THRESHOLD
                
                # Decision logic
                if prediction == 1:  # Model predicts price will go up
                    if confidence >= threshold:
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                else:  # Model predicts price will go down
                    if confidence >= threshold:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                
                # Log predictions occasionally
                if signal != 'HOLD' or np.random.random() < 0.01:
                    try:
                        feature_summary = f'RSI:{features[0][0]:.1f}, MACD:{features[0][1]:.3f}, Vol:{features[0][4]:.2f}'
                    except:
                        feature_summary = 'Features processed'
                    
                    if signal != 'HOLD':
                        log_prediction(signal, confidence, feature_summary)
                
                return signal, confidence
            
        except Exception as e:
            logger.error(f'Prediction failed: {e}')
        
        return 'HOLD', 0.5
    
    def should_retrain(self):
        """Check if models should be retrained (less frequent for stability)"""
        if self.last_training is None:
            return True
        
        # Retrain less frequently for model stability
        hours_since_training = (datetime.now() - self.last_training).total_seconds() / 3600
        return hours_since_training >= (config.MODEL_RETRAIN_HOURS * 2)  # Double the interval
    
    def get_model_performance(self):
        """Get comprehensive model performance metrics"""
        if not self.performance_history:
            return None
        
        recent_performance = self.performance_history[-5:]  # Last 5 trainings
        avg_accuracy = np.mean([p['accuracy'] for p in recent_performance])
        
        return {
            'average_accuracy': avg_accuracy,
            'best_accuracy': self.best_performance,
            'last_training': self.last_training,
            'training_count': self.training_count,
            'total_samples': len(self.cumulative_training_data.get('X', [])),
            'feature_importance': self.feature_importance,
            'improvements': len(self.model_improvements),
            'last_improvement': self.model_improvements[-1] if self.model_improvements else None
        }
    
    def _save_models(self):
        """Save the persistent model"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save as the main persistent model
            model_data = {
                'trained_models': self.trained_models,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'last_training': self.last_training,
                'performance_history': self.performance_history,
                'training_count': self.training_count,
                'cumulative_training_data': self.cumulative_training_data,
                'best_performance': self.best_performance,
                'model_improvements': self.model_improvements,
                'model_version': f'v{self.training_count}'
            }
            
            # Save main model
            with open('models/sol_persistent_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            # Also create a backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f'models/sol_model_backup_{timestamp}_v{self.training_count}.pkl'
            with open(backup_filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f'Persistent model saved (v{self.training_count}) - Best: {self.best_performance:.1%}')
            
        except Exception as e:
            logger.error(f'Failed to save persistent model: {e}')
    
    def load_models(self):
        """Load the persistent model"""
        try:
            model_file = 'models/sol_persistent_model.pkl'
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.trained_models = model_data.get('trained_models', {})
                self.scaler = model_data.get('scaler', StandardScaler())
                self.feature_importance = model_data.get('feature_importance', {})
                self.last_training = model_data.get('last_training')
                self.performance_history = model_data.get('performance_history', [])
                self.training_count = model_data.get('training_count', 0)
                self.cumulative_training_data = model_data.get('cumulative_training_data', {'X': [], 'y': []})
                self.best_performance = model_data.get('best_performance', 0.0)
                self.model_improvements = model_data.get('model_improvements', [])
                
                logger.info(f'Loaded persistent model v{self.training_count}')
                logger.info(f'Total training samples: {len(self.cumulative_training_data.get("X", []))}')
                logger.info(f'Model improvements: {len(self.model_improvements)}')
                
                return True
            
        except Exception as e:
            logger.error(f'Failed to load persistent model: {e}')
        
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
    
    def get_learning_progress(self):
        """Get information about the model's learning progress"""
        if not self.model_improvements:
            return "No improvements recorded yet"
        
        total_improvement = self.best_performance - (self.model_improvements[0]['accuracy'] - self.model_improvements[0]['improvement'])
        latest = self.model_improvements[-1]
        
        progress_info = f"""
Learning Progress:
- Total Accuracy Improvement: {total_improvement:.1%}
- Current Best: {self.best_performance:.1%}
- Total Improvements: {len(self.model_improvements)}
- Latest Improvement: {latest['improvement']:.1%} on {latest['timestamp'].strftime('%Y-%m-%d %H:%M')}
- Training Sessions: {self.training_count}
- Total Samples Learned: {len(self.cumulative_training_data.get('X', []))}
        """
        return progress_info.strip()

# Global instance - single persistent model
ml_predictor = PersistentMLPredictor()