# main.py
"""SOL trading system with ML-driven strategies"""

import asyncio
import signal
import sys
import os
from datetime import datetime, timedelta
import argparse
import pandas as pd
import numpy as np
import json

# Add project structure to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utilities.config import config
from utilities.logger import logger, log_performance
from data.data_manager import data_manager
from engine.ml_predictor import ml_predictor

def extract_features(data, index):
    """Feature extraction that works reliably"""
    try:
        if index >= len(data):
            return None
        
        current_row = data.iloc[index]
        
        feature_names = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'volume_ratio',
            'price_change_1h', 'price_change_4h', 'price_change_24h',
            'volatility', 'price_position'
        ]
        
        features = []
        for name in feature_names:
            if name in current_row and not pd.isna(current_row[name]):
                features.append(float(current_row[name]))
            else:
                features.append(0.0)
        
        # Add ratios
        close_price = float(current_row.get('close', 20.0))
        sma_20 = float(current_row.get('sma_20', close_price))
        sma_50 = float(current_row.get('sma_50', close_price))
        
        features.extend([
            close_price / sma_20,
            close_price / sma_50,
            sma_20 / sma_50,
        ])
        
        if len(features) != 13:
            return None
            
        return np.array(features).reshape(1, -1)
        
    except:
        return None

def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    return obj

class SOLTrader:
    """Main SOL trading system with ML integration"""
    
    def __init__(self):
        self.running = False
        self.last_model_training = None
        self.performance_log_interval = 300
        self.last_performance_log = datetime.now()
        
        # Trading state
        self.balance = config.INITIAL_BALANCE
        self.sol_balance = 0.0
        self.position = None
        self.trades = []
        self.max_balance = config.INITIAL_BALANCE
        self.max_drawdown = 0.0
        
        # Enhanced tracking
        self.signal_history = []
        self.trade_log_count = 0
        self.backtest_run_id = None
        
    def reset_for_new_backtest(self):
        """Reset trading state but keep the persistent model"""
        # Generate unique run ID
        self.backtest_run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Reset only trading state (NOT the ML model)
        self.balance = config.INITIAL_BALANCE
        self.sol_balance = 0.0
        self.position = None
        self.trades = []
        self.max_balance = config.INITIAL_BALANCE
        self.max_drawdown = 0.0
        
        # Reset tracking
        self.signal_history = []
        self.trade_log_count = 0
        
        # Reset random seeds for data variation
        config.reset_randomness()
        
        # DO NOT reset ML predictor - keep learning across runs
        
        logger.info(f'SOL Trader reset for backtest run: {self.backtest_run_id}')
        logger.info(f'Using persistent model v{ml_predictor.training_count} (best: {ml_predictor.best_performance:.1%})')
        
    async def start(self):
        """Start the SOL trading system"""
        try:
            logger.info(f'Starting SOL Trader in {config.MODE} mode')
            
            # Reset state for new run
            if config.MODE == 'backtest':
                self.reset_for_new_backtest()
            
            # Initialize components
            await self._initialize_system()
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._shutdown_handler)
            signal.signal(signal.SIGTERM, self._shutdown_handler)
            
            # Run based on mode
            if config.MODE == 'backtest':
                await self._run_backtest()
            else:
                await self._run_live_trading()
                
        except Exception as e:
            logger.error(f'Failed to start SOL trader: {e}')
            await self.shutdown()
    
    async def _initialize_system(self):
        """Initialize all system components"""
        # Initialize data manager
        await data_manager.initialize()
        
        # Load or fetch data based on mode
        if config.MODE == 'backtest':
            if not data_manager.load_backtest_data():
                logger.error('Failed to load backtest data')
                sys.exit(1)
        else:
            if not await data_manager.fetch_historical_data():
                logger.error('Failed to fetch historical data')
                sys.exit(1)
        
        # Calculate technical indicators
        data_manager.calculate_technical_indicators()
        
        # Initialize persistent ML predictor (always load existing model)
        ml_predictor.initialize()
        
        # Train models if needed (this will improve the existing model)
        await self._train_models_if_needed()
        
        logger.info('System initialization complete')
    
    async def _train_models_if_needed(self):
        """Train ML models if needed (continuous learning)"""
        if ml_predictor.should_retrain() or not ml_predictor.trained_models:
            logger.info('Training/improving persistent model...')
            
            X, y = data_manager.get_training_data()
            if X is not None and len(X) > 100:
                success = ml_predictor.train_models(X, y)
                if success:
                    self.last_model_training = datetime.now()
                    ml_predictor.analyze_feature_importance()
                    
                    # Log learning progress
                    progress = ml_predictor.get_learning_progress()
                    logger.info('Model training completed successfully')
                    logger.info(progress)
                else:
                    logger.error('Failed to train models')
            else:
                logger.warning('Insufficient data for ML training')
    
    async def _run_backtest(self):
        """Run backtesting mode"""
        logger.info(f'Running backtest #{self.backtest_run_id} on SOL data')
        
        data = data_manager.features
        start_idx = config.LOOKBACK_PERIODS
        
        logger.info(f'Backtesting from index {start_idx} to {len(data)} ({len(data) - start_idx} periods)')
        logger.info(f'Random seed: {config.random_seed}')
        logger.info(f'Prediction threshold: {config.PREDICTION_THRESHOLD}')
        
        # Stats tracking
        total_predictions = 0
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        
        # Trading stats
        buy_attempts = 0
        sell_attempts = 0
        holds_while_in_position = 0
        holds_while_no_position = 0
        above_threshold_buy = 0
        above_threshold_sell = 0
        
        # Backtest loop
        for i in range(start_idx, len(data)):
            
            # Extract features
            features = extract_features(data, i)
            if features is None:
                continue
            
            # Make prediction
            try:
                signal, confidence = ml_predictor.predict(features)
                total_predictions += 1
                current_price = data.iloc[i]['close']
                
                # Track signal distribution
                if signal == 'BUY':
                    buy_signals += 1
                    if confidence > config.PREDICTION_THRESHOLD:
                        above_threshold_buy += 1
                        if self.position is None:
                            buy_attempts += 1
                elif signal == 'SELL':
                    sell_signals += 1
                    if confidence > config.PREDICTION_THRESHOLD:
                        above_threshold_sell += 1
                        if self.position is not None:
                            sell_attempts += 1
                else:  # HOLD
                    hold_signals += 1
                    if self.position is not None:
                        holds_while_in_position += 1
                    else:
                        holds_while_no_position += 1
                
                # Store signal for analysis
                self.signal_history.append({
                    'index': i,
                    'signal': signal,
                    'confidence': confidence,
                    'price': current_price,
                    'has_position': self.position is not None,
                    'above_threshold': confidence > config.PREDICTION_THRESHOLD
                })
                
                # Execute trading logic
                executed = await self._process_signal_with_logging(signal, confidence, current_price, data.index[i], i)
                
            except Exception as e:
                logger.error(f'Error at index {i}: {e}')
            
            # Progress logging
            if i % 1000 == 0:
                progress = (i - start_idx) / (len(data) - start_idx) * 100
                portfolio_value = self.balance + (self.sol_balance * current_price)
                logger.info(f'Progress: {progress:.1f}% | Portfolio: ${portfolio_value:.2f} | Trades: {len(self.trades)}')
                logger.info(f'Signals - BUY:{buy_signals} SELL:{sell_signals} HOLD:{hold_signals}')
                logger.info(f'Above threshold - BUY:{above_threshold_buy} SELL:{above_threshold_sell}')
                logger.info(f'Actions - Buy attempts:{buy_attempts} Sell attempts:{sell_attempts}')
        
        # Close final position
        if self.position:
            final_price = data.iloc[-1]['close']
            await self._execute_sell(final_price, 1.0, data.index[-1], "FINAL_CLOSE")
        
        # Calculate and display results
        await self._display_backtest_results(
            total_predictions, buy_signals, sell_signals, hold_signals,
            buy_attempts, sell_attempts, holds_while_in_position, holds_while_no_position,
            above_threshold_buy, above_threshold_sell
        )
    
    async def _process_signal_with_logging(self, signal, confidence, current_price, timestamp, index):
        """Process signal with detailed logging"""
        
        # Log every 100th signal to track activity
        if index % 100 == 0 or len(self.trades) < 50:
            logger.info(f'Index {index}: {signal} conf:{confidence:.3f} price:${current_price:.4f} pos:{self.position is not None} thresh:{config.PREDICTION_THRESHOLD}')
        
        # HOLD signal logic
        if signal == 'HOLD':
            return False
        
        # Use original prediction threshold
        min_confidence = config.PREDICTION_THRESHOLD
        
        # BUY signal
        if signal == 'BUY':
            if self.position is not None:
                if index % 500 == 0:
                    logger.info(f'BUY signal ignored - already in position at ${current_price:.4f}')
                return False
            
            if confidence <= min_confidence:
                if index % 500 == 0:
                    logger.info(f'BUY signal ignored - low confidence {confidence:.3f} <= {min_confidence}')
                return False
            
            return await self._execute_buy(current_price, confidence, timestamp)
        
        # SELL signal
        if signal == 'SELL':
            if self.position is None:
                if index % 500 == 0:
                    logger.info(f'SELL signal ignored - no position at ${current_price:.4f}')
                return False
            
            if confidence <= min_confidence:
                if index % 500 == 0:
                    logger.info(f'SELL signal ignored - low confidence {confidence:.3f} <= {min_confidence}')
                return False
            
            return await self._execute_sell(current_price, confidence, timestamp, "SELL_SIGNAL")
        
        # Check stop loss/take profit
        if self.position is not None:
            should_sell = False
            sell_reason = ""
            
            if current_price <= self.position['stop_loss']:
                should_sell = True
                sell_reason = "STOP_LOSS"
            elif current_price >= self.position['take_profit']:
                should_sell = True
                sell_reason = "TAKE_PROFIT"
            
            if should_sell:
                logger.info(f'{sell_reason} triggered at ${current_price:.4f}')
                return await self._execute_sell(current_price, confidence, timestamp, sell_reason)
        
        return False
    
    async def _execute_buy(self, price, confidence, timestamp):
        """Execute buy order"""
        try:
            # Calculate position size
            risk_amount = self.balance * config.MAX_POSITION_SIZE * confidence
            amount = risk_amount / price
            
            if amount * price >= config.MIN_TRADE_AMOUNT:
                cost = amount * price * (1 + config.TRADING_FEE)
                self.balance -= cost
                self.sol_balance += amount
                
                # Set stop loss and take profit
                stop_loss = price * (1 - config.STOP_LOSS_PCT)
                take_profit = price * (1 + config.TAKE_PROFIT_PCT)
                
                self.position = {
                    'entry_price': price,
                    'amount': amount,
                    'entry_time': timestamp,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
                # Convert timestamp to string
                timestamp_str = serialize_datetime(timestamp)
                
                self.trades.append({
                    'timestamp': timestamp_str,
                    'action': 'BUY',
                    'price': price,
                    'amount': amount,
                    'balance': self.balance,
                    'confidence': confidence,
                    'run_id': self.backtest_run_id
                })
                
                self.trade_log_count += 1
                
                logger.info(f'BUY #{self.trade_log_count}: ${price:.4f} | Amount: {amount:.4f} SOL | Confidence: {confidence:.3f} | Balance: ${self.balance:.2f}')
                
                return True
        
        except Exception as e:
            logger.error(f'Buy execution failed: {e}')
        
        return False
    
    async def _execute_sell(self, price, confidence, timestamp, reason="SELL_SIGNAL"):
        """Execute sell order"""
        try:
            if not self.position:
                return False
            
            amount = self.position['amount']
            revenue = amount * price * (1 - config.TRADING_FEE)
            self.balance += revenue
            self.sol_balance -= amount
            
            # Calculate PnL
            cost = amount * self.position['entry_price'] * (1 + config.TRADING_FEE)
            pnl = revenue - cost
            
            # Update max balance and drawdown
            if self.balance > self.max_balance:
                self.max_balance = self.balance
            
            current_drawdown = (self.max_balance - self.balance) / self.max_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Convert timestamp to string
            timestamp_str = serialize_datetime(timestamp)
            
            self.trades.append({
                'timestamp': timestamp_str,
                'action': 'SELL',
                'price': price,
                'amount': amount,
                'balance': self.balance,
                'confidence': confidence,
                'pnl': pnl,
                'reason': reason,
                'run_id': self.backtest_run_id
            })
            
            self.trade_log_count += 1
            
            logger.info(f'SELL #{self.trade_log_count}: ${price:.4f} | PnL: ${pnl:.2f} | Reason: {reason} | Balance: ${self.balance:.2f}')
            
            self.position = None
            return True
        
        except Exception as e:
            logger.error(f'Sell execution failed: {e}')
        
        return False
    
    async def _display_backtest_results(self, total_predictions, buy_signals, sell_signals, hold_signals,
                                       buy_attempts, sell_attempts, holds_while_in_position, holds_while_no_position,
                                       above_threshold_buy, above_threshold_sell):
        """Display comprehensive backtest results"""
        
        # Calculate metrics
        initial_balance = config.INITIAL_BALANCE
        final_portfolio_value = self.balance + (self.sol_balance * data_manager.features.iloc[-1]['close'])
        total_return = (final_portfolio_value - initial_balance) / initial_balance
        
        # Calculate win rate
        profitable_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        total_executed_trades = len([t for t in self.trades if 'pnl' in t])
        win_rate = profitable_trades / total_executed_trades if total_executed_trades > 0 else 0
        
        # Calculate total PnL
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        
        # Calculate average holding time
        holding_times = []
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy_time_str = self.trades[i]['timestamp']
                sell_time_str = self.trades[i + 1]['timestamp']
                buy_time = datetime.fromisoformat(buy_time_str)
                sell_time = datetime.fromisoformat(sell_time_str)
                holding_time = (sell_time - buy_time).total_seconds() / 3600
                holding_times.append(holding_time)
        
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Calculate Sharpe ratio
        if len(self.trades) > 1:
            returns = [t.get('pnl', 0) / initial_balance for t in self.trades if 'pnl' in t]
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 0
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        start_date = data_manager.features.index[0].strftime('%Y-%m-%d')
        end_date = data_manager.features.index[-1].strftime('%Y-%m-%d')
        
        print("\n" + "="*70)
        print(f"SOL TRADING SYSTEM - BACKTEST #{self.backtest_run_id}")
        print("="*70)
        print(f"Random Seed: {config.random_seed}")
        print(f"Model Version: v{ml_predictor.training_count} (Best: {ml_predictor.best_performance:.1%})")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"SOL Holdings: {self.sol_balance:.6f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"Total Return: {total_return:.1%}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print("-" * 70)
        print("SIGNAL ANALYSIS:")
        print(f"Total Predictions: {total_predictions:,}")
        print(f"BUY Signals: {buy_signals:,} ({buy_signals/total_predictions:.1%})")
        print(f"SELL Signals: {sell_signals:,} ({sell_signals/total_predictions:.1%})")
        print(f"HOLD Signals: {hold_signals:,} ({hold_signals/total_predictions:.1%})")
        print(f"Above Threshold BUY: {above_threshold_buy:,}")
        print(f"Above Threshold SELL: {above_threshold_sell:,}")
        print(f"Prediction Threshold: {config.PREDICTION_THRESHOLD}")
        print("-" * 70)
        print("TRADING BEHAVIOR:")
        print(f"Buy Attempts: {buy_attempts}")
        print(f"Sell Attempts: {sell_attempts}")
        print(f"Holds while in position: {holds_while_in_position}")
        print(f"Holds while no position: {holds_while_no_position}")
        print(f"Actual Trades Executed: {len(self.trades)}")
        print(f"Trade Execution Rate: {len(self.trades)/(buy_attempts+sell_attempts):.1%}" if (buy_attempts+sell_attempts) > 0 else "Trade Execution Rate: N/A")
        print("-" * 70)
        print("PERFORMANCE METRICS:")
        print(f"Profitable Trades: {profitable_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Holding Time: {avg_holding_time:.1f} hours")
        print(f"Max Drawdown: {self.max_drawdown:.1%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print("="*70)
        
        # Log final performance
        log_performance(len(self.trades), win_rate, total_pnl, self.max_drawdown)
        
        # Save results
        await self._save_backtest_results({
            'run_id': self.backtest_run_id,
            'random_seed': config.random_seed,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'current_balance': self.balance,
            'sol_balance': self.sol_balance,
            'signal_analysis': {
                'total_predictions': total_predictions,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'buy_attempts': buy_attempts,
                'sell_attempts': sell_attempts,
                'holds_while_in_position': holds_while_in_position,
                'holds_while_no_position': holds_while_no_position,
                'above_threshold_buy': above_threshold_buy,
                'above_threshold_sell': above_threshold_sell
            },
            'avg_holding_time_hours': avg_holding_time
        })
    
    async def _save_backtest_results(self, results):
        """Save backtest results to file"""
        try:
            os.makedirs('results', exist_ok=True)
            
            # Prepare trades data
            trades_data = []
            for trade in self.trades:
                trade_dict = {}
                for key, value in trade.items():
                    trade_dict[key] = serialize_datetime(value)
                trades_data.append(trade_dict)
            
            # Prepare signal history
            signal_data = []
            for signal in self.signal_history[-1000:]:
                signal_dict = {}
                for key, value in signal.items():
                    signal_dict[key] = serialize_datetime(value)
                signal_data.append(signal_dict)
            
            # Get ML performance
            ml_performance = ml_predictor.get_model_performance()
            if ml_performance and 'last_training' in ml_performance:
                ml_performance['last_training'] = serialize_datetime(ml_performance['last_training'])
            
            backtest_results = {
                'config': {
                    'initial_balance': config.INITIAL_BALANCE,
                    'max_position_size': config.MAX_POSITION_SIZE,
                    'stop_loss_pct': config.STOP_LOSS_PCT,
                    'take_profit_pct': config.TAKE_PROFIT_PCT,
                    'prediction_threshold': config.PREDICTION_THRESHOLD
                },
                'results': results,
                'trades': trades_data,
                'signal_history': signal_data,
                'ml_performance': ml_performance
            }
            
            filename = f'results/sol_backtest_{self.backtest_run_id}.json'
            
            with open(filename, 'w') as f:
                json.dump(backtest_results, f, indent=2, default=str)
            
            logger.info(f'Backtest results saved to {filename}')
            
        except Exception as e:
            logger.error(f'Failed to save backtest results: {e}')
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f'Received signal {signum}, initiating shutdown...')
        self.running = False
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info('Shutting down SOL trader...')
        self.running = False
        
        # Close any open positions
        if self.position:
            try:
                current_price = await data_manager.get_current_price()
                if current_price:
                    await self._execute_sell(current_price, 1.0, datetime.now(), "SHUTDOWN")
                    logger.info('Closed open position during shutdown')
            except Exception as e:
                logger.error(f'Failed to close position during shutdown: {e}')
        
        # Final performance log
        try:
            current_price = await data_manager.get_current_price()
            if current_price:
                await self._log_performance(current_price)
        except:
            pass
        
        logger.info('SOL trader shutdown complete')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SOL Trading System')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--balance', type=float, default=10000, 
                       help='Initial balance for paper/backtest trading')
    parser.add_argument('--threshold', type=float, default=0.55,
                       help='Prediction confidence threshold for trading')
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Update config based on arguments
    config.MODE = args.mode
    config.INITIAL_BALANCE = args.balance
    config.PREDICTION_THRESHOLD = args.threshold
    
    print("SOL Trading System")
    print(f"Mode: {config.MODE.upper()}")
    print(f"Initial Balance: ${config.INITIAL_BALANCE:.2f}")
    print(f"Prediction Threshold: {config.PREDICTION_THRESHOLD}")
    print("-" * 40)
    
    if config.MODE in ['paper', 'live'] and not config.BINANCE_API_KEY:
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        print("For testnet keys, visit: https://testnet.binance.vision/")
        return
    
    # Single run
    trader = SOLTrader()
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        logger.info('Keyboard interrupt received')
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
    finally:
        await trader.shutdown()

if __name__ == '__main__':
    asyncio.run(main())