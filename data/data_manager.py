# data_manager.py
"""Data management for SOL trading"""

import pandas as pd
import numpy as np
import ccxt
import asyncio
from datetime import datetime, timedelta
import pickle
import os
from utilities.config import config
from utilities.logger import logger

class DataManager:
    """Manages SOL market data and features"""
    
    def __init__(self):
        self.exchange = None
        self.data = pd.DataFrame()
        self.features = pd.DataFrame()
        
    async def initialize(self):
        """Initialize exchange connection"""
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'sandbox': config.SANDBOX_MODE,
            'enableRateLimit': True,
        })
        
        if config.MODE != 'backtest':
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
        """Load data for backtesting"""
        try:
            if os.path.exists(file_path):
                self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                logger.info(f'Loaded {len(self.data)} backtest data points')
                return True
            else:
                # Generate sample data if file doesn't exist
                self._generate_sample_data()
                return True
        except Exception as e:
            logger.error(f'Failed to load backtest data: {e}')
            return False
    
    def _generate_sample_data(self):
        """Generate sample SOL data for backtesting"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
        
        # Simulate SOL price movement
        price = 20.0  # Starting price
        prices = []
        
        for i in range(len(dates)):
            # Add trend and volatility
            trend = 0.0001 * np.sin(i / 100)  # Long-term trend
            volatility = np.random.normal(0, 0.02)  # Random volatility
            price *= (1 + trend + volatility)
            prices.append(price)
        
        # Create OHLCV data
        self.data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Save for future use
        os.makedirs('data', exist_ok=True)
        self.data.to_csv('data/sol_data.csv')
        logger.info(f'Generated {len(self.data)} sample data points')
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators for SOL"""
        if self.data.empty:
            return
        
        df = self.data.copy()
        
        # Price-based indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
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
        
        self.features = df.dropna()
        logger.info(f'Calculated technical indicators for {len(self.features)} periods')
    
    def get_latest_features(self):
        """Get latest feature vector for prediction"""
        if self.features.empty:
            return None
        
        latest = self.features.iloc[-1]
        feature_names = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'volume_ratio',
            'price_change_1h', 'price_change_4h', 'price_change_24h',
            'volatility', 'price_position'
        ]
        
        features = []
        for name in feature_names:
            if name in latest:
                features.append(latest[name])
            else:
                features.append(0.0)
        
        # Add price ratios
        features.extend([
            latest['close'] / latest['sma_20'] if latest['sma_20'] > 0 else 1.0,
            latest['close'] / latest['sma_50'] if latest['sma_50'] > 0 else 1.0,
            latest['sma_20'] / latest['sma_50'] if latest['sma_50'] > 0 else 1.0,
        ])
        
        return np.array(features)
    
    def get_training_data(self, lookback=None):
        """Get training data for ML model"""
        if lookback is None:
            lookback = config.LOOKBACK_PERIODS
        
        if len(self.features) < lookback + 1:
            return None, None
        
        # Prepare features
        feature_names = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'volume_ratio',
            'price_change_1h', 'price_change_4h', 'price_change_24h',
            'volatility', 'price_position'
        ]
        
        X = []
        y = []
        
        for i in range(lookback, len(self.features)):
            # Features from current and past periods
            current_features = []
            
            for j in range(lookback):
                row = self.features.iloc[i - lookback + j]
                period_features = []
                
                for name in feature_names:
                    if name in row:
                        period_features.append(row[name])
                    else:
                        period_features.append(0.0)
                
                current_features.extend(period_features)
            
            # Add current ratios
            current_row = self.features.iloc[i]
            current_features.extend([
                current_row['close'] / current_row['sma_20'] if current_row['sma_20'] > 0 else 1.0,
                current_row['close'] / current_row['sma_50'] if current_row['sma_50'] > 0 else 1.0,
                current_row['sma_20'] / current_row['sma_50'] if current_row['sma_50'] > 0 else 1.0,
            ])
            
            X.append(current_features)
            
            # Target: price change in next period
            if i < len(self.features) - 1:
                future_price = self.features.iloc[i + 1]['close']
                current_price = self.features.iloc[i]['close']
                price_change = (future_price - current_price) / current_price
                
                # Convert to classification: 1 for up, 0 for down
                y.append(1 if price_change > 0.001 else 0)  # 0.1% threshold
            else:
                y.append(0)
        
        return np.array(X), np.array(y)
    
    async def get_current_price(self):
        """Get current SOL price"""
        try:
            if config.MODE == 'backtest':
                return self.features.iloc[-1]['close']
            else:
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