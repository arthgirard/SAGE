# config.py
"""Configuration for SOL trading system"""

import os
import numpy as np
import random
from datetime import datetime

# Remove fixed random seeds to allow variation
# os.environ['NUMEXPR_MAX_THREADS'] = '16'

class Config:
    # SOL trading symbol (fixed)
    SYMBOL = 'SOL/USDT'
    
    # Exchange settings
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    
    # Trading modes
    MODE = os.getenv('TRADING_MODE', 'paper')  # 'backtest', 'paper', 'live'
    SANDBOX_MODE = MODE != 'live'
    
    # Trading parameters
    BASE_CURRENCY = 'USDT'
    INITIAL_BALANCE = 10000.0
    MIN_TRADE_AMOUNT = 10.0
    TRADING_FEE = 0.001
    
    # Risk management
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio per trade
    STOP_LOSS_PCT = 0.03     # 3% stop loss
    TAKE_PROFIT_PCT = 0.06   # 6% take profit
    MAX_DAILY_LOSS = 0.05    # 5% max daily loss
    
    # ML parameters
    LOOKBACK_PERIODS = 50
    FEATURES_COUNT = 20
    MODEL_RETRAIN_HOURS = 24
    PREDICTION_THRESHOLD = 0.55
    
    # Data settings
    TIMEFRAME = '1h'
    DATA_LIMIT = 1000
    
    def __init__(self):
        # Initialize with random seed based on current time to ensure variation
        self.random_seed = int(datetime.now().timestamp() * 1000000) % 2**32
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def reset_randomness(self):
        """Reset random seeds for new backtest run"""
        self.random_seed = int(datetime.now().timestamp() * 1000000) % 2**32
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

config = Config()