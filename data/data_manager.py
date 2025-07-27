# data/data_manager.py
"""Data management for SOL trading"""

import pandas as pd
import numpy as np
import ccxt
import asyncio
from datetime import datetime, timedelta
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.config import config
from utilities.logger import logger

class DataManager:
    """Manages SOL market data and features"""
    
    def __init__(self):
        self.exchange = None
        self.data = pd.DataFrame()
        self.features = pd.DataFrame()
        self.data_generation_count = 0
        
    async def initialize(self):
        """Initialize exchange connection"""
        if config.MODE == 'backtest':
            # No need for exchange connection in backtest mode
            logger.info('Data manager initialized for backtest mode')
            return
            
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'sandbox': config.SANDBOX_MODE,
            'enableRateLimit': True,
        })
        
        await self.exchange.load_markets()
        logger.info('Data manager initialized')
    
    async def fetch_historical_data(self, days_back=30):
        """Fetch historical SOL data"""
        try:
            since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            ohlcv = await self.exchange.fetch_ohlcv(
                config.SYMBOL, 
                config.TIMEFRAME, 
                since=since,
                limit=config.DATA_LIMIT
            )
            
            self.data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')
            self.data.set_index('timestamp', inplace=True)
            
            logger.info(f'Fetched {len(self.data)} historical data points')
            return True
            
        except Exception as e:
            logger.error(f'Failed to fetch historical data: {e}')
            return False
    
    def load_backtest_data(self, file_path='data/sol_data.csv'):
        """Load data for backtesting with minimal variation"""
        try:
            if os.path.exists(file_path):
                # Load existing data with minimal variation for consistent model training
                self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Add very small random variations to create slight differences between runs
                # But keep them minimal so the model can learn consistent patterns
                price_variation = np.random.normal(0, 0.0005, len(self.data))  # 0.05% variation
                volume_variation = np.random.normal(1, 0.02, len(self.data))   # 2% volume variation
                
                self.data['close'] *= (1 + price_variation)
                self.data['open'] *= (1 + price_variation * 0.8)
                self.data['high'] *= (1 + np.maximum(price_variation, 0) * 1.2)
                self.data['low'] *= (1 + np.minimum(price_variation, 0) * 1.2)
                self.data['volume'] *= np.maximum(volume_variation, 0.1)
                
                logger.info(f'Loaded {len(self.data)} backtest data points with minimal variations')
                return True
            else:
                # Generate new sample data
                self._generate_sample_data()
                return True
        except Exception as e:
            logger.error(f'Failed to load backtest data: {e}')
            return False
    
    def _generate_sample_data(self):
        """Generate sample SOL data with randomness"""
        # Reset random seed for this generation to ensure variation
        generation_seed = int(datetime.now().timestamp() * 1000000) % 2**32 + self.data_generation_count
        np.random.seed(generation_seed)
        
        self.data_generation_count += 1
        
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
        
        # Simulate SOL price movement with more randomness
        base_price = np.random.uniform(15.0, 25.0)  # Random starting price
        trend_strength = np.random.uniform(-0.00005, 0.00005)  # Random trend
        volatility_base = np.random.uniform(0.015, 0.025)  # Random base volatility
        
        prices = []
        price = base_price
        
        for i in range(len(dates)):
            # Add multiple sources of randomness
            trend = trend_strength * np.sin(i / np.random.uniform(80, 120))  # Random cycle length
            volatility = volatility_base * (1 + 0.5 * np.sin(i / np.random.uniform(200, 300)))  # Time-varying volatility
            shock = np.random.normal(0, 0.001) if np.random.random() < 0.05 else 0  # 5% chance of shock
            daily_drift = np.random.normal(0, 0.0005)  # Daily drift
            
            price_change = trend + np.random.normal(0, volatility) + shock + daily_drift
            price *= (1 + price_change)
            
            # Ensure price stays reasonable
            price = max(5.0, min(100.0, price))
            prices.append(price)
        
        # Create OHLCV data with realistic relationships
        opens = prices.copy()
        closes = prices.copy()
        
        highs = []
        lows = []
        volumes = []
        
        for i, price in enumerate(prices):
            # Generate high/low with correlation to volatility
            intraday_vol = np.random.uniform(0.005, 0.03)
            high = price * (1 + abs(np.random.normal(0, intraday_vol)))
            low = price * (1 - abs(np.random.normal(0, intraday_vol)))
            
            # Ensure OHLC relationships are valid
            high = max(high, price)
            low = min(low, price)
            
            highs.append(high)
            lows.append(low)
            
            # Generate volume with some correlation to price movement
            base_volume = np.random.uniform(1000000, 5000000)
            if i > 0:
                price_change_impact = abs(price - prices[i-1]) / prices[i-1] * 10
                volume_multiplier = 1 + price_change_impact
            else:
                volume_multiplier = 1
            
            volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
            volumes.append(volume)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
        
        # Save for potential reuse
        os.makedirs('data', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/sol_data_{timestamp}.csv'
        self.data.to_csv(filename)
        
        # Also save as default
        self.data.to_csv('data/sol_data.csv')
        
        logger.info(f'Generated {len(self.data)} sample data points with seed {generation_seed}')
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators for SOL"""
        if self.data.empty:
            return
        
        df = self.data.copy()
        
        # Add small random noise to break deterministic calculations
        noise_factor = 1e-8
        df['close'] += np.random.normal(0, noise_factor, len(df))
        
        # Price-based indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI with slight randomization
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Add small random component to RSI to break exact repetition
        df['rsi'] += np.random.normal(0, 0.1, len(df))
        df['rsi'] = np.clip(df['rsi'], 0, 100)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change_1h'] = df['close'].pct_change(1)
        df['price_change_4h'] = df['close'].pct_change(4)
        df['price_change_24h'] = df['close'].pct_change(24)
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Support/Resistance levels
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # Fill NaN values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN, fill with neutral values
        df['rsi'] = df['rsi'].fillna(50.0)
        df['macd'] = df['macd'].fillna(0.0)
        df['macd_histogram'] = df['macd_histogram'].fillna(0.0)
        df['bb_position'] = df['bb_position'].fillna(0.5)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        df['price_change_1h'] = df['price_change_1h'].fillna(0.0)
        df['price_change_4h'] = df['price_change_4h'].fillna(0.0)
        df['price_change_24h'] = df['price_change_24h'].fillna(0.0)
        df['volatility'] = df['volatility'].fillna(0.02)
        df['price_position'] = df['price_position'].fillna(0.5)
        
        self.features = df
        logger.info(f'Calculated technical indicators for {len(self.features)} periods')
    
    def get_latest_features(self):
        """Get latest feature vector for prediction"""
        if self.features.empty:
            return None
        
        latest = self.features.iloc[-1]
        return self._extract_simple_features(latest)
    
    def _extract_simple_features(self, row):
        """Extract simple feature vector from a data row"""
        feature_names = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'volume_ratio',
            'price_change_1h', 'price_change_4h', 'price_change_24h',
            'volatility', 'price_position'
        ]
        
        features = []
        for name in feature_names:
            if name in row and not pd.isna(row[name]):
                features.append(float(row[name]))
            else:
                features.append(0.0)
        
        # Add price ratios
        close_price = row.get('close', 20.0)
        sma_20 = row.get('sma_20', close_price)
        sma_50 = row.get('sma_50', close_price)
        
        features.extend([
            close_price / sma_20 if sma_20 > 0 else 1.0,
            close_price / sma_50 if sma_50 > 0 else 1.0,
            sma_20 / sma_50 if sma_50 > 0 else 1.0,
        ])
        
        return np.array(features)
    
    def get_training_data(self, lookback=None):
        """Get training data for ML model with consistent approach"""
        if self.features.empty:
            return None, None
        
        # Use consistent training data selection for stable model learning
        feature_names = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'volume_ratio',
            'price_change_1h', 'price_change_4h', 'price_change_24h',
            'volatility', 'price_position'
        ]
        
        X = []
        y = []
        
        # Use consistent starting point for stable learning
        start_idx = 50
        for i in range(start_idx, len(self.features) - 1):
            current_row = self.features.iloc[i]
            next_row = self.features.iloc[i + 1]
            
            # Extract features for current row
            features = self._extract_simple_features(current_row)
            X.append(features)
            
            # Target: price direction in next period
            current_price = current_row['close']
            next_price = next_row['close']
            price_change = (next_price - current_price) / current_price
            
            # Use consistent threshold for stable learning
            threshold = 0.001  # 0.1% fixed threshold
            y.append(1 if price_change > threshold else 0)
        
        if len(X) == 0:
            return None, None
        
        return np.array(X), np.array(y)
    
    async def get_current_price(self):
        """Get current SOL price"""
        try:
            if config.MODE == 'backtest':
                if not self.features.empty:
                    return self.features.iloc[-1]['close']
                else:
                    return 20.0  # Default price for testing
            else:
                if self.exchange is None:
                    return None
                ticker = await self.exchange.fetch_ticker(config.SYMBOL)
                return ticker['last']
        except Exception as e:
            logger.error(f'Failed to get current price: {e}')
            return None
    
    def save_data(self, file_path='data/sol_features.pkl'):
        """Save processed data"""
        os.makedirs('data', exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'features': self.features
            }, f)
        
    def load_data(self, file_path='data/sol_features.pkl'):
        """Load processed data"""
        try:
            with open(file_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.data = saved_data['data']
                self.features = saved_data['features']
                return True
        except:
            return False

# Global instance
data_manager = DataManager()