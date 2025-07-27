# config.py
"""Configuration for SOL trading system"""

import os

# Suppress NumExpr threads warning
os.environ['NUMEXPR_MAX_THREADS'] = '16'

class Config:
    # Trading symbol
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
    LOOKBACK_PERIODS = 50  # Reduced from 100 to ensure enough data
    FEATURES_COUNT = 20
    MODEL_RETRAIN_HOURS = 24
    PREDICTION_THRESHOLD = 0.55  # Increased to make bot more selective and hold more
    
    # Data settings
    TIMEFRAME = '1h'
    DATA_LIMIT = 1000

config = Config()