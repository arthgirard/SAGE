# logger.py
"""Logging system for SOL trading"""

import logging
import os
import sys
from datetime import datetime

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f'logs/sol_trader_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
    ]
)

logger = logging.getLogger('SOL_TRADER')

def log_trade(action, price, amount, balance, pnl=None, confidence=None):
    """Log trading activity"""
    pnl_str = f' | PnL: ${pnl:.2f}' if pnl is not None else ''
    conf_str = f' | Confidence: {confidence:.2f}' if confidence is not None else ''
    logger.info(f'TRADE | {action} | ${price:.4f} | {amount:.4f} SOL | Balance: ${balance:.2f}{pnl_str}{conf_str}')

def log_prediction(signal, confidence, features_summary):
    """Log ML prediction"""
    logger.info(f'PREDICTION : {signal} | Confidence: {confidence:.3f} | Features: {features_summary}')

def log_performance(total_trades, win_rate, total_pnl, max_drawdown):
    """Log performance metrics"""
    logger.info(f'PERFORMANCE | Trades: {total_trades} | Win Rate: {win_rate:.1%} | PnL: ${total_pnl:.2f} | Max DD: {max_drawdown:.1%}')

def log_backtest_result(start_date, end_date, total_return, sharpe_ratio, max_dd):
    """Log backtest results"""
    logger.info(f'BACKTEST | {start_date} to {end_date} | Return: {total_return:.1%} | Sharpe: {sharpe_ratio:.2f} | Max DD: {max_dd:.1%}')