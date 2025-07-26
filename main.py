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

class SOLTrader:
    """Main SOL trading system with ML integration"""
    
    def __init__(self):
        self.running = False
        self.last_model_training = None
        self.performance_log_interval = 300  # 5 minutes
        self.last_performance_log = datetime.now()
        
        # Trading state
        self.balance = config.INITIAL_BALANCE
        self.sol_balance = 0.0
        self.position = None
        self.trades = []
        self.max_balance = config.INITIAL_BALANCE
        self.max_drawdown = 0.0
        
    async def start(self):
        """Start the SOL trading system"""
        try:
            logger.info(f'Starting SOL Trader in {config.MODE} mode')
            
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
        
        # Initialize ML predictor
        if not ml_predictor.load_models():
            logger.info('No existing models found, will train new ones')
        
        # Train models if needed
        await self._train_models_if_needed()
        
        logger.info('System initialization complete')
    
    async def _train_models_if_needed(self):
        """Train ML models if needed"""
        if ml_predictor.should_retrain() or not ml_predictor.trained_models:
            logger.info('Training ML models...')
            
            X, y = data_manager.get_training_data()
            if X is not None and len(X) > 100:
                success = ml_predictor.train_models(X, y)
                if success:
                    self.last_model_training = datetime.now()
                    ml_predictor.analyze_feature_importance()
                    logger.info('ML models trained successfully')
                else:
                    logger.error('Failed to train ML models')
            else:
                logger.warning('Insufficient data for ML training')
    
    async def _run_backtest(self):
        """Run backtesting mode"""
        logger.info('Running complete backtest on SOL data')
        
        data = data_manager.features
        start_idx = config.LOOKBACK_PERIODS
        
        logger.info(f'Backtesting from index {start_idx} to {len(data)} ({len(data) - start_idx} periods)')
        
        # Stats tracking
        total_predictions = 0
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        
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
                
                if signal == 'BUY':
                    buy_signals += 1
                elif signal == 'SELL':
                    sell_signals += 1
                else:
                    hold_signals += 1
                
                # Execute trading logic
                current_price = data.iloc[i]['close']
                await self._process_signal(signal, confidence, current_price, data.index[i])
                
            except Exception as e:
                logger.error(f'Error at index {i}: {e}')
            
            # Progress logging
            if i % 1000 == 0:
                progress = (i - start_idx) / (len(data) - start_idx) * 100
                portfolio_value = self.balance + (self.sol_balance * current_price)
                logger.info(f'Progress: {progress:.1f}% | Portfolio: ${portfolio_value:.2f} | Trades: {len(self.trades)}')
        
        # Close final position
        if self.position:
            final_price = data.iloc[-1]['close']
            await self._execute_sell(final_price, 1.0, data.index[-1], "FINAL_CLOSE")
        
        # Calculate and display results
        await self._display_backtest_results(total_predictions, buy_signals, sell_signals, hold_signals)
    
    async def _process_signal(self, signal, confidence, current_price, timestamp):
        """Process trading signal"""
        
        # Buy signal
        if signal == 'BUY' and confidence > config.PREDICTION_THRESHOLD and self.position is None:
            await self._execute_buy(current_price, confidence, timestamp)
        
        # Check stop loss/take profit or sell signal
        elif self.position is not None:
            should_sell = False
            sell_reason = ""
            
            # Check stop loss/take profit
            if current_price <= self.position['stop_loss']:
                should_sell = True
                sell_reason = "STOP_LOSS"
            elif current_price >= self.position['take_profit']:
                should_sell = True
                sell_reason = "TAKE_PROFIT"
            elif signal == 'SELL' and confidence > config.PREDICTION_THRESHOLD:
                should_sell = True
                sell_reason = "SELL_SIGNAL"
            
            if should_sell:
                await self._execute_sell(current_price, confidence, timestamp, sell_reason)
    
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
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': price,
                    'amount': amount,
                    'balance': self.balance,
                    'confidence': confidence
                })
                
                if len(self.trades) <= 20:  # Log first 20 trades
                    logger.info(f'BUY: ${price:.4f} | Amount: {amount:.4f} SOL | Confidence: {confidence:.3f}')
                
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
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': price,
                'amount': amount,
                'balance': self.balance,
                'confidence': confidence,
                'pnl': pnl,
                'reason': reason
            })
            
            if len(self.trades) <= 40:  # Log first 40 trades
                logger.info(f'SELL: ${price:.4f} | PnL: ${pnl:.2f} | Reason: {reason}')
            
            self.position = None
            return True
        
        except Exception as e:
            logger.error(f'Sell execution failed: {e}')
        
        return False
    
    async def _display_backtest_results(self, total_predictions, buy_signals, sell_signals, hold_signals):
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
        
        # Calculate Sharpe ratio (simplified)
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
        
        # Display results
        start_date = data_manager.features.index[0].strftime('%Y-%m-%d')
        end_date = data_manager.features.index[-1].strftime('%Y-%m-%d')
        
        print("\n" + "="*60)
        print("SAGE - BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"SOL Holdings: {self.sol_balance:.6f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"Total Return: {total_return:.1%}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print("-" * 60)
        print(f"Total Predictions: {total_predictions:,}")
        print(f"BUY Signals: {buy_signals:,}")
        print(f"SELL Signals: {sell_signals:,}")
        print(f"HOLD Signals: {hold_signals:,}")
        print(f"Trades Executed: {len(self.trades)}")
        print(f"Profitable Trades: {profitable_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Max Drawdown: {self.max_drawdown:.1%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print("="*60)
        
        # Log final performance
        log_performance(len(self.trades), win_rate, total_pnl, self.max_drawdown)
        
        # Save results
        await self._save_backtest_results({
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'current_balance': self.balance,
            'sol_balance': self.sol_balance
        })
    
    async def _run_live_trading(self):
        """Run live/paper trading mode"""
        logger.info(f'Starting {config.MODE} trading loop')
        self.running = True
        
        loop_count = 0
        
        while self.running:
            try:
                loop_start = datetime.now()
                
                # Update market data (for non-backtest modes)
                await self._update_market_data()
                
                # Retrain models periodically
                if ml_predictor.should_retrain():
                    await self._train_models_if_needed()
                
                # Get current price
                current_price = await data_manager.get_current_price()
                if current_price is None:
                    logger.warning('Failed to get current price')
                    await asyncio.sleep(30)
                    continue
                
                # Get ML prediction
                features = data_manager.get_latest_features()
                if features is not None:
                    features_reshaped = features.reshape(1, -1) if features.ndim == 1 else features
                    signal, confidence = ml_predictor.predict(features_reshaped)
                    
                    # Execute signal
                    if signal != 'HOLD':
                        await self._process_signal(signal, confidence, current_price, datetime.now())
                
                # Log performance periodically
                if (datetime.now() - self.last_performance_log).seconds >= self.performance_log_interval:
                    await self._log_performance(current_price)
                    self.last_performance_log = datetime.now()
                
                loop_count += 1
                
                # Sleep between iterations
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, 60 - loop_duration)  # 1-minute intervals
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f'Error in trading loop: {e}')
                await asyncio.sleep(30)
    
    async def _update_market_data(self):
        """Update market data for live/paper trading"""
        try:
            # Fetch recent data
            if await data_manager.fetch_historical_data(days_back=2):
                data_manager.calculate_technical_indicators()
            
        except Exception as e:
            logger.error(f'Failed to update market data: {e}')
    
    async def _log_performance(self, current_price: float):
        """Log current performance"""
        try:
            portfolio_value = self.balance + (self.sol_balance * current_price)
            
            profitable_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
            total_executed_trades = len([t for t in self.trades if 'pnl' in t])
            win_rate = profitable_trades / total_executed_trades if total_executed_trades > 0 else 0
            total_pnl = sum(t.get('pnl', 0) for t in self.trades)
            
            log_performance(len(self.trades), win_rate, total_pnl, self.max_drawdown)
            
            logger.info(f'Portfolio Value: ${portfolio_value:.2f} | SOL Price: ${current_price:.4f}')
            
            # Log ML model performance
            ml_performance = ml_predictor.get_model_performance()
            if ml_performance:
                logger.info(f'Model Accuracy: {ml_performance["average_accuracy"]:.3f} | '
                           f'Trainings: {ml_performance["training_count"]}')
            
        except Exception as e:
            logger.error(f'Error logging performance: {e}')
    
    async def _save_backtest_results(self, results):
        """Save backtest results to file"""
        try:
            os.makedirs('results', exist_ok=True)
            
            # Add trades data
            trades_data = []
            for trade in self.trades:
                trade_dict = dict(trade)
                if 'timestamp' in trade_dict:
                    trade_dict['timestamp'] = trade_dict['timestamp'].isoformat()
                trades_data.append(trade_dict)
            
            backtest_results = {
                'config': {
                    'symbol': config.SYMBOL,
                    'initial_balance': config.INITIAL_BALANCE,
                    'max_position_size': config.MAX_POSITION_SIZE,
                    'stop_loss_pct': config.STOP_LOSS_PCT,
                    'take_profit_pct': config.TAKE_PROFIT_PCT,
                    'prediction_threshold': config.PREDICTION_THRESHOLD
                },
                'results': results,
                'trades': trades_data
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'results/sol_backtest_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(backtest_results, f, indent=2)
            
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
    parser = argparse.ArgumentParser(description='Solana Analysis and Guidance Engine')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--symbol', default='SOL/USDT', help='Trading symbol')
    parser.add_argument('--balance', type=float, default=10000, 
                       help='Initial balance for paper/backtest trading')
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Update config based on arguments
    config.MODE = args.mode
    config.SYMBOL = args.symbol
    config.INITIAL_BALANCE = args.balance
    
    print("🟢 SAGE online")
    print(f"Mode: {config.MODE.upper()}")
    print(f"Symbol: {config.SYMBOL}")
    print(f"Initial Balance: ${config.INITIAL_BALANCE:.2f}")
    print("-" * 40)
    
    if config.MODE in ['paper', 'live'] and not config.BINANCE_API_KEY:
        print("⚠️  Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        print("For testnet keys, visit: https://testnet.binance.vision/")
        return
    
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