# main.py
"""SOL trading system with ML-driven strategies"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
import argparse

from utilities.config import config
from utilities.logger import logger, log_performance
from data.data_manager import data_manager
from engine.ml_predictor import ml_predictor
from engine.trading_engine import trading_engine

class SOLTrader:
    """Main SOL trading system with ML integration"""
    
    def __init__(self):
        self.running = False
        self.last_model_training = None
        self.performance_log_interval = 300  # 5 minutes
        self.last_performance_log = datetime.now()
        
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
        
        # Initialize trading engine
        await trading_engine.initialize()
        
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
        logger.info('Running backtest mode')
        
        # Run backtest
        results = trading_engine.run_backtest()
        
        if results:
            logger.info('Backtest completed successfully')
            
            # Display results
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            print(f"Total Trades: {results['total_trades']}")
            print(f"Win Rate: {results['win_rate']:.1%}")
            print(f"Total Return: {results['total_return']:.1%}")
            print(f"Total PnL: ${results['total_pnl']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.1%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Final Balance: ${results['current_balance']:.2f}")
            print(f"SOL Holdings: {results['sol_balance']:.4f}")
            print("="*50)
            
            # Save results
            self._save_backtest_results(results)
        else:
            logger.error('Backtest failed')
    
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
                
                # Check stop loss/take profit first
                if trading_engine.position:
                    if await trading_engine.check_stop_loss_take_profit(current_price):
                        logger.info('Position closed by stop loss/take profit')
                
                # Get ML prediction
                features = data_manager.get_latest_features()
                if features is not None:
                    signal, confidence = ml_predictor.predict(features)
                    
                    # Execute signal
                    if signal != 'HOLD':
                        await trading_engine.execute_signal(signal, confidence, current_price)
                
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
            metrics = trading_engine.get_performance_metrics()
            portfolio_value = trading_engine.get_portfolio_value(current_price)
            
            log_performance(
                metrics.get('total_trades', 0),
                metrics.get('win_rate', 0),
                metrics.get('total_pnl', 0),
                metrics.get('max_drawdown', 0)
            )
            
            logger.info(f'Portfolio Value: ${portfolio_value:.2f} | SOL Price: ${current_price:.4f}')
            
            # Log ML model performance
            ml_performance = ml_predictor.get_model_performance()
            if ml_performance:
                logger.info(f'Model Accuracy: {ml_performance["average_accuracy"]:.3f} | '
                           f'Trainings: {ml_performance["training_count"]}')
            
        except Exception as e:
            logger.error(f'Error logging performance: {e}')
    
    def _save_backtest_results(self, results):
        """Save backtest results to file"""
        try:
            import json
            import os
            
            os.makedirs('results', exist_ok=True)
            
            # Add trades data
            trades_data = []
            for trade in trading_engine.trades:
                trades_data.append({
                    'timestamp': trade.timestamp.isoformat(),
                    'action': trade.action,
                    'price': trade.price,
                    'amount': trade.amount,
                    'pnl': trade.pnl,
                    'confidence': trade.confidence
                })
            
            # Add equity curve
            equity_data = []
            for point in trading_engine.equity_curve:
                equity_data.append({
                    'timestamp': point['timestamp'].isoformat(),
                    'equity': point['equity'],
                    'balance': point['balance'],
                    'sol_value': point['sol_value']
                })
            
            backtest_results = {
                'config': {
                    'symbol': config.SYMBOL,
                    'initial_balance': config.INITIAL_BALANCE,
                    'max_position_size': config.MAX_POSITION_SIZE,
                    'stop_loss_pct': config.STOP_LOSS_PCT,
                    'take_profit_pct': config.TAKE_PROFIT_PCT
                },
                'results': results,
                'trades': trades_data,
                'equity_curve': equity_data
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'results/backtest_{timestamp}.json'
            
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
        if trading_engine.position:
            try:
                current_price = await data_manager.get_current_price()
                if current_price:
                    await trading_engine._execute_sell(current_price, 1.0)
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
        
        # Close connections
        await trading_engine.close()
        
        logger.info('SOL trader shutdown complete')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SOL Trading System')
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
    
    print("🟢 SOL Trading System")
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