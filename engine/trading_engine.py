# engine/trading_engine.py
"""Trading engine for SOL with backtest, paper, and live trading"""

import ccxt
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.config import config
from utilities.logger import logger, log_trade, log_performance, log_backtest_result

@dataclass
class Trade:
    timestamp: datetime
    action: str  # 'BUY' or 'SELL'
    price: float
    amount: float
    balance_before: float
    balance_after: float
    confidence: float = 0.0
    pnl: float = 0.0

@dataclass
class Position:
    entry_price: float
    amount: float
    entry_time: datetime
    stop_loss: float
    take_profit: float

class TradingEngine:
    """Unified trading engine for all modes"""
    
    def __init__(self):
        self.exchange = None
        self.balance = config.INITIAL_BALANCE
        self.sol_balance = 0.0
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.max_balance = config.INITIAL_BALANCE
        self.max_drawdown = 0.0
        
        # Backtest specific
        self.backtest_index = 0
        self.backtest_data = None
    
    async def initialize(self):
        """Initialize trading engine based on mode"""
        if config.MODE == 'live':
            await self._initialize_live()
        elif config.MODE == 'paper':
            await self._initialize_paper()
        else:  # backtest
            self._initialize_backtest()
        
        logger.info(f'Trading engine initialized in {config.MODE} mode')
    
    async def _initialize_live(self):
        """Initialize for live trading"""
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        await self.exchange.load_markets()
        
        # Get actual balance
        balance_data = await self.exchange.fetch_balance()
        self.balance = balance_data.get('USDT', {}).get('free', 0.0)
        self.sol_balance = balance_data.get('SOL', {}).get('free', 0.0)
        
        logger.info(f'Live trading initialized - USDT: ${self.balance:.2f}, SOL: {self.sol_balance:.4f}')
    
    async def _initialize_paper(self):
        """Initialize for paper trading"""
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'sandbox': True,
            'enableRateLimit': True,
        })
        
        await self.exchange.load_markets()
        logger.info(f'Paper trading initialized - Balance: ${self.balance:.2f}')
    
    def _initialize_backtest(self):
        """Initialize for backtesting"""
        # Import here to avoid circular imports
        from data.data_manager import data_manager
        self.backtest_data = data_manager.features.copy()
        self.backtest_index = config.LOOKBACK_PERIODS
        logger.info(f'Backtest initialized - {len(self.backtest_data)} data points')
    
    async def execute_signal(self, signal: str, confidence: float, current_price: float):
        """Execute trading signal"""
        if confidence < config.PREDICTION_THRESHOLD:
            return False
        
        if signal == 'BUY' and self.position is None:
            return await self._execute_buy(current_price, confidence)
        elif signal == 'SELL' and self.position is not None:
            return await self._execute_sell(current_price, confidence)
        
        return False
    
    async def _execute_buy(self, price: float, confidence: float):
        """Execute buy order"""
        try:
            # Calculate position size based on confidence and risk management
            risk_amount = self.balance * config.MAX_POSITION_SIZE * confidence
            amount = min(risk_amount / price, (self.balance - config.MIN_TRADE_AMOUNT) / price)
            
            if amount * price < config.MIN_TRADE_AMOUNT:
                logger.warning(f'Trade amount too small: ${amount * price:.2f}')
                return False
            
            # Execute based on mode
            if config.MODE == 'live':
                success = await self._execute_live_buy(amount, price)
            elif config.MODE == 'paper':
                success = await self._execute_paper_buy(amount, price)
            else:  # backtest
                success = self._execute_backtest_buy(amount, price)
            
            if success:
                # Set stop loss and take profit
                stop_loss = price * (1 - config.STOP_LOSS_PCT)
                take_profit = price * (1 + config.TAKE_PROFIT_PCT)
                
                self.position = Position(
                    entry_price=price,
                    amount=amount,
                    entry_time=datetime.utcnow() if config.MODE != 'backtest' else self.backtest_data.index[self.backtest_index],
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                # Update balances
                cost = amount * price * (1 + config.TRADING_FEE)
                balance_before = self.balance
                self.balance -= cost
                self.sol_balance += amount
                
                # Record trade
                trade = Trade(
                    timestamp=self.position.entry_time,
                    action='BUY',
                    price=price,
                    amount=amount,
                    balance_before=balance_before,
                    balance_after=self.balance,
                    confidence=confidence
                )
                self.trades.append(trade)
                self.total_trades += 1
                
                log_trade('BUY', price, amount, self.balance, confidence=confidence)
                return True
            
        except Exception as e:
            logger.error(f'Buy execution failed: {e}')
        
        return False
    
    async def _execute_sell(self, price: float, confidence: float):
        """Execute sell order"""
        try:
            if not self.position:
                return False
            
            amount = self.position.amount
            
            # Execute based on mode
            if config.MODE == 'live':
                success = await self._execute_live_sell(amount, price)
            elif config.MODE == 'paper':
                success = await self._execute_paper_sell(amount, price)
            else:  # backtest
                success = self._execute_backtest_sell(amount, price)
            
            if success:
                # Calculate PnL
                revenue = amount * price * (1 - config.TRADING_FEE)
                cost = amount * self.position.entry_price * (1 + config.TRADING_FEE)
                pnl = revenue - cost
                
                # Update balances
                balance_before = self.balance
                self.balance += revenue
                self.sol_balance -= amount
                
                # Track performance
                if pnl > 0:
                    self.winning_trades += 1
                
                # Update max balance and drawdown
                if self.balance > self.max_balance:
                    self.max_balance = self.balance
                
                current_drawdown = (self.max_balance - self.balance) / self.max_balance
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
                
                # Record trade
                trade = Trade(
                    timestamp=datetime.utcnow() if config.MODE != 'backtest' else self.backtest_data.index[self.backtest_index],
                    action='SELL',
                    price=price,
                    amount=amount,
                    balance_before=balance_before,
                    balance_after=self.balance,
                    confidence=confidence,
                    pnl=pnl
                )
                self.trades.append(trade)
                self.total_trades += 1
                
                # Record equity point
                total_value = self.balance + (self.sol_balance * price)
                self.equity_curve.append({
                    'timestamp': trade.timestamp,
                    'equity': total_value,
                    'balance': self.balance,
                    'sol_value': self.sol_balance * price
                })
                
                log_trade('SELL', price, amount, self.balance, pnl=pnl, confidence=confidence)
                
                # Clear position
                self.position = None
                return True
            
        except Exception as e:
            logger.error(f'Sell execution failed: {e}')
        
        return False
    
    async def _execute_live_buy(self, amount: float, price: float):
        """Execute live buy order"""
        try:
            order = await self.exchange.create_market_order(config.SYMBOL, 'buy', amount)
            logger.info(f'Live buy order executed: {order["id"]}')
            return True
        except Exception as e:
            logger.error(f'Live buy failed: {e}')
            return False
    
    async def _execute_live_sell(self, amount: float, price: float):
        """Execute live sell order"""
        try:
            order = await self.exchange.create_market_order(config.SYMBOL, 'sell', amount)
            logger.info(f'Live sell order executed: {order["id"]}')
            return True
        except Exception as e:
            logger.error(f'Live sell failed: {e}')
            return False
    
    async def _execute_paper_buy(self, amount: float, price: float):
        """Execute paper buy order"""
        logger.info(f'Paper buy executed - would buy {amount:.4f} SOL at ${price:.4f}')
        return True
    
    async def _execute_paper_sell(self, amount: float, price: float):
        """Execute paper sell order"""
        logger.info(f'Paper sell executed - would sell {amount:.4f} SOL at ${price:.4f}')
        return True
    
    def _execute_backtest_buy(self, amount: float, price: float):
        """Execute backtest buy order"""
        return True
    
    def _execute_backtest_sell(self, amount: float, price: float):
        """Execute backtest sell order"""
        return True
    
    async def check_stop_loss_take_profit(self, current_price: float):
        """Check stop loss and take profit levels"""
        if not self.position:
            return False
        
        # Check stop loss
        if current_price <= self.position.stop_loss:
            logger.warning(f'Stop loss triggered at ${current_price:.4f}')
            return await self._execute_sell(current_price, 1.0)
        
        # Check take profit
        if current_price >= self.position.take_profit:
            logger.info(f'Take profit triggered at ${current_price:.4f}')
            return await self._execute_sell(current_price, 1.0)
        
        return False
    
    def advance_backtest(self):
        """Advance backtest to next period"""
        if config.MODE == 'backtest':
            self.backtest_index += 1
            return self.backtest_index < len(self.backtest_data)
        return True
    
    def get_current_backtest_data(self):
        """Get current backtest data point"""
        if config.MODE == 'backtest' and self.backtest_index < len(self.backtest_data):
            return self.backtest_data.iloc[self.backtest_index]
        return None
    
    def get_portfolio_value(self, current_price: float):
        """Get total portfolio value"""
        return self.balance + (self.sol_balance * current_price)
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        # Calculate returns
        initial_value = config.INITIAL_BALANCE
        current_value = self.balance + (self.sol_balance * (self.trades[-1].price if self.trades else 20.0))
        total_return = (current_value - initial_value) / initial_value
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': total_return,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'current_balance': self.balance,
                'sol_balance': self.sol_balance
            }
        
        win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0
        total_pnl = sum(trade.pnl for trade in self.trades if trade.pnl != 0)
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]['equity']
                curr_equity = self.equity_curve[i]['equity']
                ret = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                returns.append(ret)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 0
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'current_balance': self.balance,
            'sol_balance': self.sol_balance
        }
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run complete backtest"""
        if config.MODE != 'backtest':
            logger.error('Backtest can only be run in backtest mode')
            return None
        
        logger.info('Starting backtest...')
        
        # Import here to avoid circular imports
        from data.data_manager import data_manager
        from engine.ml_predictor import ml_predictor
        
        # Filter data by date range if provided
        backtest_data = self.backtest_data.copy()
        if start_date:
            backtest_data = backtest_data[backtest_data.index >= start_date]
        if end_date:
            backtest_data = backtest_data[backtest_data.index <= end_date]
        
        if len(backtest_data) < config.LOOKBACK_PERIODS + 10:
            logger.error('Insufficient backtest data')
            return None
        
        # Reset state
        self.balance = config.INITIAL_BALANCE
        self.sol_balance = 0.0
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.total_trades = 0
        self.winning_trades = 0
        self.max_balance = config.INITIAL_BALANCE
        self.max_drawdown = 0.0
        
        # Get training data for initial model training
        X, y = data_manager.get_training_data()
        if X is not None and len(X) > 100:
            ml_predictor.train_models(X, y)
            logger.info('Models trained for backtest')
        else:
            logger.warning('Insufficient data for ML training')
            return None
        
        # Run backtest loop
        processed_trades = 0
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_sum = 0
        prediction_count = 0
        
        logger.info(f'Starting backtest loop from index {config.LOOKBACK_PERIODS} to {len(backtest_data)}')
        
        for i in range(config.LOOKBACK_PERIODS, len(backtest_data)):
            self.backtest_index = i
            current_data = backtest_data.iloc[i]
            current_price = current_data['close']
            
            # Progress logging with signal stats
            if i % 1000 == 0:
                progress = (i - config.LOOKBACK_PERIODS) / (len(backtest_data) - config.LOOKBACK_PERIODS) * 100
                avg_conf = confidence_sum / prediction_count if prediction_count > 0 else 0
                logger.info(f'Progress: {progress:.1f}% - Price: ${current_price:.4f} - Avg Conf: {avg_conf:.3f}')
                logger.info(f'Signals: BUY:{signal_counts["BUY"]} SELL:{signal_counts["SELL"]} HOLD:{signal_counts["HOLD"]}')
            
            # Get features for prediction
            features = self._get_backtest_features(backtest_data, i)
            if features is not None:
                # Make prediction
                signal, confidence = ml_predictor.predict(features)
                
                # Track signal statistics
                signal_counts[signal] += 1
                confidence_sum += confidence
                prediction_count += 1
                
                # Debug first few predictions
                if i < config.LOOKBACK_PERIODS + 10:
                    logger.info(f'Prediction {i}: {signal} (conf: {confidence:.3f}, threshold: {config.PREDICTION_THRESHOLD})')
                
                # Check stop loss/take profit first
                if self.position:
                    stop_triggered = self._check_stop_loss_take_profit_sync(current_price)
                    if stop_triggered:
                        processed_trades += 1
                        logger.info(f'Position closed by stop/profit at ${current_price:.4f}')
                        continue
                
                # Execute signal
                executed = self._execute_signal_sync(signal, confidence, current_price)
                if executed:
                    processed_trades += 1
                    logger.info(f'Trade executed: {signal} at ${current_price:.4f} with confidence {confidence:.3f}')
            else:
                if i < config.LOOKBACK_PERIODS + 5:
                    logger.warning(f'No features available for index {i}')
            
            # Record equity curve every 24 periods
            if i % 24 == 0:
                total_value = self.get_portfolio_value(current_price)
                self.equity_curve.append({
                    'timestamp': current_data.name,
                    'equity': total_value,
                    'balance': self.balance,
                    'sol_value': self.sol_balance * current_price
                })
        
        # Final statistics
        logger.info(f'Final signal distribution - BUY: {signal_counts["BUY"]}, SELL: {signal_counts["SELL"]}, HOLD: {signal_counts["HOLD"]}')
        avg_confidence = confidence_sum / prediction_count if prediction_count > 0 else 0
        logger.info(f'Average prediction confidence: {avg_confidence:.3f}, Threshold: {config.PREDICTION_THRESHOLD}')
        
        # Count signals above threshold
        above_threshold = sum(1 for signal in ['BUY', 'SELL'] if signal_counts[signal] > 0)
        logger.info(f'Predictions made: {prediction_count}, Trades attempted: {processed_trades}')
        
        # Close final position if exists
        if self.position:
            final_price = backtest_data.iloc[-1]['close']
            self._execute_sell_sync(final_price, 1.0)
            logger.info(f'Final position closed at ${final_price:.4f}')
        
        logger.info(f'Backtest completed - Processed {processed_trades} trading events')
        
        # Calculate final metrics
        metrics = self.get_performance_metrics()
        
        # Log backtest results
        start_str = backtest_data.index[0].strftime('%Y-%m-%d')
        end_str = backtest_data.index[-1].strftime('%Y-%m-%d')
        
        log_backtest_result(
            start_str, end_str,
            metrics['total_return'],
            metrics['sharpe_ratio'],
            metrics['max_drawdown']
        )
        
        log_performance(
            metrics['total_trades'],
            metrics['win_rate'],
            metrics['total_pnl'],
            metrics['max_drawdown']
        )
        
        return metrics
    
    def _get_backtest_features(self, data, index):
        """Get features for backtest prediction"""
        if index < config.LOOKBACK_PERIODS:
            return None
        
        if index >= len(data):
            return None
        
        try:
            # Import here to avoid circular imports
            from data.data_manager import data_manager
            
            # Get current data point
            current_row = data.iloc[index]
            
            # Use data_manager's feature extraction method
            features = data_manager._extract_simple_features(current_row)
            
            # Ensure proper shape for model input
            if features is not None and len(features) > 0:
                return features.reshape(1, -1)
            
            return None
            
        except Exception as e:
            logger.error(f'Error extracting features at index {index}: {e}')
            return None
    
    def _check_stop_loss_take_profit_sync(self, current_price: float):
        """Synchronous version for backtesting"""
        if not self.position:
            return False
        
        # Check stop loss
        if current_price <= self.position.stop_loss:
            logger.warning(f'Stop loss triggered at ${current_price:.4f}')
            return self._execute_sell_sync(current_price, 1.0)
        
        # Check take profit
        if current_price >= self.position.take_profit:
            logger.info(f'Take profit triggered at ${current_price:.4f}')
            return self._execute_sell_sync(current_price, 1.0)
        
        return False
    
    def _execute_signal_sync(self, signal: str, confidence: float, current_price: float):
        """Synchronous signal execution for backtesting"""
        # Debug logging for first few attempts
        if self.backtest_index < config.LOOKBACK_PERIODS + 10:
            logger.info(f'Signal attempt: {signal}, confidence: {confidence:.3f}, threshold: {config.PREDICTION_THRESHOLD}, has_position: {self.position is not None}')
        
        if confidence < config.PREDICTION_THRESHOLD:
            return False
        
        if signal == 'BUY' and self.position is None:
            return self._execute_buy_sync(current_price, confidence)
        elif signal == 'SELL' and self.position is not None:
            return self._execute_sell_sync(current_price, confidence)
        
        return False
    
    def _execute_buy_sync(self, price: float, confidence: float):
        """Synchronous buy execution for backtesting"""
        try:
            # Calculate position size
            risk_amount = self.balance * config.MAX_POSITION_SIZE * confidence
            amount = min(risk_amount / price, (self.balance - config.MIN_TRADE_AMOUNT) / price)
            
            if amount * price < config.MIN_TRADE_AMOUNT:
                return False
            
            # Set stop loss and take profit
            stop_loss = price * (1 - config.STOP_LOSS_PCT)
            take_profit = price * (1 + config.TAKE_PROFIT_PCT)
            
            self.position = Position(
                entry_price=price,
                amount=amount,
                entry_time=self.backtest_data.index[self.backtest_index],
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Update balances
            cost = amount * price * (1 + config.TRADING_FEE)
            balance_before = self.balance
            self.balance -= cost
            self.sol_balance += amount
            
            # Record trade
            trade = Trade(
                timestamp=self.position.entry_time,
                action='BUY',
                price=price,
                amount=amount,
                balance_before=balance_before,
                balance_after=self.balance,
                confidence=confidence
            )
            self.trades.append(trade)
            self.total_trades += 1
            
            return True
            
        except Exception as e:
            logger.error(f'Buy execution failed: {e}')
            return False
    
    def _execute_sell_sync(self, price: float, confidence: float):
        """Synchronous sell execution for backtesting"""
        try:
            if not self.position:
                return False
            
            amount = self.position.amount
            
            # Calculate PnL
            revenue = amount * price * (1 - config.TRADING_FEE)
            cost = amount * self.position.entry_price * (1 + config.TRADING_FEE)
            pnl = revenue - cost
            
            # Update balances
            balance_before = self.balance
            self.balance += revenue
            self.sol_balance -= amount
            
            # Track performance
            if pnl > 0:
                self.winning_trades += 1
            
            # Update max balance and drawdown
            if self.balance > self.max_balance:
                self.max_balance = self.balance
            
            current_drawdown = (self.max_balance - self.balance) / self.max_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Record trade
            trade = Trade(
                timestamp=self.backtest_data.index[self.backtest_index],
                action='SELL',
                price=price,
                amount=amount,
                balance_before=balance_before,
                balance_after=self.balance,
                confidence=confidence,
                pnl=pnl
            )
            self.trades.append(trade)
            self.total_trades += 1
            
            # Clear position
            self.position = None
            return True
            
        except Exception as e:
            logger.error(f'Sell execution failed: {e}')
            return False
    
    async def close(self):
        """Close trading engine"""
        if self.exchange:
            await self.exchange.close()
            logger.info('Trading engine closed')

# Global instance
trading_engine = TradingEngine()