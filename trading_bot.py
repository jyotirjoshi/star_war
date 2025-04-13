import pandas as pd
import numpy as np
import logging
import os
import time
import json
import threading
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Union, Optional
from abc import ABC, abstractmethod
import config
from data_handler import DataHandler
from strategies import TradingStrategy, create_strategies
from rl_agent import RLTrainer, DQNAgent, TradingEnv
from backtester import Backtester, BacktestResult

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.RESULTS_DIR, "trading_bot.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")


class TradingMode(ABC):
    """Abstract base class for trading modes"""

    @abstractmethod
    def execute(self, data: pd.DataFrame, **kwargs):
        """Execute the trading mode"""
        pass


class BacktestMode(TradingMode):
    """Backtest mode for historical testing"""

    def execute(self, data: pd.DataFrame, **kwargs):
        """Run backtesting on historical data"""
        strategies = kwargs.get('strategies', None)
        if strategies is None:
            strategies = create_strategies()

        # Get backtest parameters
        commission = kwargs.get('commission', config.COMMISSION_PER_CONTRACT)
        slippage = kwargs.get('slippage', config.SLIPPAGE_TICKS)
        position_size = kwargs.get('position_size', config.MAX_POSITION_SIZE)
        use_stop_loss = kwargs.get('use_stop_loss', True)
        stop_loss_pct = kwargs.get('stop_loss_pct', config.STOP_LOSS_PCT)
        use_take_profit = kwargs.get('use_take_profit', False)
        take_profit_pct = kwargs.get('take_profit_pct', config.TAKE_PROFIT_PCT)
        use_trailing_stop = kwargs.get('use_trailing_stop', False)
        trailing_stop_pct = kwargs.get('trailing_stop_pct', config.TRAILING_STOP_PCT)

        # Create backtester
        backtester = Backtester(data)

        if isinstance(strategies, dict):
            # Run backtest for all strategies
            results = {}
            for name, strategy in strategies.items():
                logger.info(f"Running backtest for {name} strategy")
                result = backtester.run_backtest(
                    strategy=strategy,
                    commission=commission,
                    slippage=slippage,
                    position_size=position_size,
                    use_stop_loss=use_stop_loss,
                    stop_loss_pct=stop_loss_pct,
                    use_take_profit=use_take_profit,
                    take_profit_pct=take_profit_pct,
                    use_trailing_stop=use_trailing_stop,
                    trailing_stop_pct=trailing_stop_pct
                )
                results[name] = result

            # Compare strategies
            comparison = backtester.compare_strategies(results)

            # Plot comparison
            comparison_path = os.path.join(config.RESULTS_DIR, "strategy_comparison.png")
            backtester.plot_comparison(results, comparison_path)

            # Save detailed results for each strategy
            for name, result in results.items():
                result_folder = os.path.join(config.RESULTS_DIR, "backtest_results", name.replace(" ", "_").lower())
                result.save_results(result_folder)

            return results
        else:
            # Run backtest for a single strategy
            logger.info(
                f"Running backtest for {strategies.name if hasattr(strategies, 'name') else 'provided'} strategy")
            result = backtester.run_backtest(
                strategy=strategies,
                commission=commission,
                slippage=slippage,
                position_size=position_size,
                use_stop_loss=use_stop_loss,
                stop_loss_pct=stop_loss_pct,
                use_take_profit=use_take_profit,
                take_profit_pct=take_profit_pct,
                use_trailing_stop=use_trailing_stop,
                trailing_stop_pct=trailing_stop_pct
            )

            # Save results
            result_folder = os.path.join(config.RESULTS_DIR, "backtest_results",
                                         strategies.name.replace(" ", "_").lower() if hasattr(strategies,
                                                                                              'name') else "custom_strategy")
            result.save_results(result_folder)

            return result


class TrainingMode(TradingMode):
    """Training mode for RL agent"""

    def execute(self, data: pd.DataFrame, **kwargs):
        """Train RL agent on historical data"""
        # Get training parameters
        episodes = kwargs.get('episodes', config.RL_EPISODES)
        test_size = kwargs.get('test_size', 0.2)
        target_update_freq = kwargs.get('target_update_freq', 10)
        save_freq = kwargs.get('save_freq', 100)

        # Create trainer
        trainer = RLTrainer(data, test_size=test_size)

        # Train agent
        logger.info(f"Training RL agent for {episodes} episodes")
        training_results = trainer.train(episodes=episodes, target_update_freq=target_update_freq, save_freq=save_freq)

        # Evaluate agent
        best_model_path = os.path.join(config.MODELS_DIR, "dqn_trading_best.h5")
        eval_results = trainer.evaluate(best_model_path)

        # Plot results
        trainer.plot_results(training_results, eval_results)

        return {
            'training_results': training_results,
            'eval_results': eval_results,
            'model_path': best_model_path
        }


class PaperTradingMode(TradingMode):
    """Paper trading mode for simulation"""

    def __init__(self):
        self.data_handler = DataHandler()
        self.position = 0
        self.balance = config.PAPER_TRADING_BALANCE
        self.trades = []
        self.entry_price = None
        self.entry_time = None
        self.strategy_signals = {}
        self.is_running = False
        self.stop_event = threading.Event()

    def execute(self, data: pd.DataFrame = None, **kwargs):
        """Run paper trading simulation"""
        # Get parameters
        strategies = kwargs.get('strategies', None)
        if strategies is None:
            strategies = create_strategies()

        commission = kwargs.get('commission', config.COMMISSION_PER_CONTRACT)
        slippage = kwargs.get('slippage', config.SLIPPAGE_TICKS)
        position_size = kwargs.get('position_size', config.MAX_POSITION_SIZE)
        use_stop_loss = kwargs.get('use_stop_loss', True)
        stop_loss_pct = kwargs.get('stop_loss_pct', config.STOP_LOSS_PCT)
        use_take_profit = kwargs.get('use_take_profit', False)
        take_profit_pct = kwargs.get('take_profit_pct', config.TAKE_PROFIT_PCT)
        use_trailing_stop = kwargs.get('use_trailing_stop', False)
        trailing_stop_pct = kwargs.get('trailing_stop_pct', config.TRAILING_STOP_PCT)

        # Start the paper trading loop
        self.is_running = True

        try:
            # For simulation, we'll use historical data and process it sequentially
            if data is not None:
                # Use provided data for simulation
                logger.info(f"Starting paper trading simulation with provided data of length {len(data)}")

                # Calculate signals for all strategies
                for name, strategy in strategies.items():
                    self.strategy_signals[name] = strategy.generate_signals(data)

                # Simulate trading day by day
                for i in range(len(data)):
                    if self.stop_event.is_set():
                        logger.info("Paper trading stopped by user")
                        break

                    # Get current data point
                    current_time = data.index[i]
                    current_row = data.iloc[i]

                    # Combine signals from all strategies (simple majority vote)
                    combined_signal = 0
                    for name, signals in self.strategy_signals.items():
                        if current_time in signals.index:
                            weight = config.STRATEGY_WEIGHTS.get(name, 1.0)
                            combined_signal += signals.loc[current_time] * weight

                    # Determine trade action based on combined signal
                    action = 1 if combined_signal > 0.2 else (-1 if combined_signal < -0.2 else 0)

                    # Execute trade
                    self._execute_trade(current_time, current_row, action, commission, slippage, position_size,
                                        use_stop_loss, stop_loss_pct, use_take_profit, take_profit_pct,
                                        use_trailing_stop, trailing_stop_pct)

                    # Simulate delay
                    time.sleep(0.01)  # Fast simulation

                logger.info("Paper trading simulation completed")

            else:
                # Use live data for paper trading
                logger.info("Starting paper trading with live data")

                # Start live data stream
                self.data_handler.start_live_data_stream(self._on_live_data)

                # Keep the main thread alive
                while self.is_running and not self.stop_event.is_set():
                    time.sleep(1)

                # Stop live data stream
                self.data_handler.stop_live_data_stream()

        except Exception as e:
            logger.error(f"Error in paper trading: {e}")
        finally:
            self.is_running = False

        # Return trading results
        return {
            'initial_balance': config.PAPER_TRADING_BALANCE,
            'final_balance': self.balance,
            'total_trades': len(self.trades),
            'position': self.position,
            'trades': self.trades
        }

    def _on_live_data(self, tick_data):
        """Process live tick data"""
        if not self.is_running or self.stop_event.is_set():
            return

        try:
            # Process the incoming tick
            current_time = tick_data['time']

            # For live trading, we need to generate signals on the fly
            # This is simplified for the example
            strategies = create_strategies()
            combined_signal = 0

            for name, strategy in strategies.items():
                # We'd need a way to incrementally update signals for live data
                # For now, we'll use a simple approximation
                weight = config.STRATEGY_WEIGHTS.get(name, 1.0)

                # Generate signal for this tick (simplified)
                # In a real implementation, you'd accumulate data and generate signals based on sufficient history
                if hasattr(strategy, 'generate_signal_for_tick'):
                    signal = strategy.generate_signal_for_tick(tick_data)
                    combined_signal += signal * weight

            # Determine trade action based on combined signal
            action = 1 if combined_signal > 0.2 else (-1 if combined_signal < -0.2 else 0)

            # Execute trade
            self._execute_trade(
                current_time, tick_data, action,
                config.COMMISSION_PER_CONTRACT, config.SLIPPAGE_TICKS, config.MAX_POSITION_SIZE,
                True, config.STOP_LOSS_PCT, False, config.TAKE_PROFIT_PCT,
                False, config.TRAILING_STOP_PCT
            )

        except Exception as e:
            logger.error(f"Error processing live tick: {e}")

    def _execute_trade(self, timestamp, data, action, commission, slippage, position_size,
                       use_stop_loss, stop_loss_pct, use_take_profit, take_profit_pct,
                       use_trailing_stop, trailing_stop_pct):
        """Execute a trade based on action"""
        current_price = data['close']

        # Check for stop loss, take profit, or trailing stop
        if self.position != 0 and self.entry_price is not None:
            unrealized_pnl_pct = (current_price - self.entry_price) * self.position / self.entry_price * 100

            # Check stop loss
            if use_stop_loss and ((self.position > 0 and unrealized_pnl_pct <= -stop_loss_pct) or
                                  (self.position < 0 and unrealized_pnl_pct <= -stop_loss_pct)):
                # Close position due to stop loss
                exit_price = current_price - (slippage * self.position / abs(self.position))
                pnl = (exit_price - self.entry_price) * self.position * position_size * 20  # 20 is NQ multiplier

                self.balance += pnl - (commission * position_size)

                self.trades.append({
                    'entry_time': self.entry_time,
                    'entry_price': self.entry_price,
                    'exit_time': timestamp,
                    'exit_price': exit_price,
                    'position': self.position,
                    'pnl': pnl,
                    'reason': 'Stop Loss'
                })

                logger.info(f"Stop loss triggered at {timestamp}: Closed position at {exit_price}, PnL: ${pnl:.2f}")

                self.position = 0
                self.entry_price = None
                self.entry_time = None

                return

            # Check take profit
            if use_take_profit and ((self.position > 0 and unrealized_pnl_pct >= take_profit_pct) or
                                    (self.position < 0 and unrealized_pnl_pct >= take_profit_pct)):
                # Close position due to take profit
                exit_price = current_price - (slippage * self.position / abs(self.position))
                pnl = (exit_price - self.entry_price) * self.position * position_size * 20  # 20 is NQ multiplier

                self.balance += pnl - (commission * position_size)

                self.trades.append({
                    'entry_time': self.entry_time,
                    'entry_price': self.entry_price,
                    'exit_time': timestamp,
                    'exit_price': exit_price,
                    'position': self.position,
                    'pnl': pnl,
                    'reason': 'Take Profit'
                })

                logger.info(f"Take profit triggered at {timestamp}: Closed position at {exit_price}, PnL: ${pnl:.2f}")

                self.position = 0
                self.entry_price = None
                self.entry_time = None

                return

            # Trailing stop implementation would go here

        # Process new signals
        if action == 1 and self.position <= 0:  # Buy signal
            # Close existing short position if any
            if self.position < 0:
                exit_price = current_price + slippage
                pnl = (self.entry_price - exit_price) * abs(self.position) * position_size * 20  # 20 is NQ multiplier

                self.balance += pnl - (commission * position_size)

                self.trades.append({
                    'entry_time': self.entry_time,
                    'entry_price': self.entry_price,
                    'exit_time': timestamp,
                    'exit_price': exit_price,
                    'position': self.position,
                    'pnl': pnl,
                    'reason': 'Exit Short'
                })

                logger.info(f"Closed short position at {timestamp}: Exit price {exit_price}, PnL: ${pnl:.2f}")

                self.position = 0
                self.entry_price = None
                self.entry_time = None

            # Enter new long position
            entry_price = current_price + slippage

            # Check if enough balance
            position_cost = entry_price * position_size * 20  # 20 is NQ multiplier
            margin_required = position_cost * 0.1  # Assuming 10% margin requirement

            if self.balance >= margin_required:
                self.position = position_size
                self.entry_price = entry_price
                self.entry_time = timestamp
                self.balance -= commission * position_size

                logger.info(
                    f"Entered long position at {timestamp}: Entry price {entry_price}, Position: {self.position}")
            else:
                logger.warning(
                    f"Insufficient funds to enter long position: Required ${margin_required:.2f}, Available: ${self.balance:.2f}")

        elif action == -1 and self.position >= 0:  # Sell signal
            # Close existing long position if any
            if self.position > 0:
                exit_price = current_price - slippage
                pnl = (exit_price - self.entry_price) * self.position * position_size * 20  # 20 is NQ multiplier

                self.balance += pnl - (commission * position_size)

                self.trades.append({
                    'entry_time': self.entry_time,
                    'entry_price': self.entry_price,
                    'exit_time': timestamp,
                    'exit_price': exit_price,
                    'position': self.position,
                    'pnl': pnl,
                    'reason': 'Exit Long'
                })

                logger.info(f"Closed long position at {timestamp}: Exit price {exit_price}, PnL: ${pnl:.2f}")

                self.position = 0
                self.entry_price = None
                self.entry_time = None

            # Enter new short position
            entry_price = current_price - slippage

            # Check if enough balance
            position_cost = entry_price * position_size * 20  # 20 is NQ multiplier
            margin_required = position_cost * 0.1  # Assuming 10% margin requirement

            if self.balance >= margin_required:
                self.position = -position_size
                self.entry_price = entry_price
                self.entry_time = timestamp
                self.balance -= commission * position_size

                logger.info(
                    f"Entered short position at {timestamp}: Entry price {entry_price}, Position: {self.position}")
            else:
                logger.warning(
                    f"Insufficient funds to enter short position: Required ${margin_required:.2f}, Available: ${self.balance:.2f}")

    def stop(self):
        """Stop paper trading"""
        self.stop_event.set()
        logger.info("Paper trading stop signal received")


class LiveTradingMode(TradingMode):
    """Live trading mode with real money (WARNING: USE WITH CAUTION)"""

    def __init__(self):
        self.data_handler = DataHandler()

        # Placeholder for broker API
        self.broker = None
        self.is_running = False
        self.stop_event = threading.Event()

    def execute(self, data: pd.DataFrame = None, **kwargs):
        """Run live trading"""
        # WARNING: This is just a placeholder implementation
        # Actual live trading would require integration with a broker API

        logger.warning("LIVE TRADING MODE - USE WITH EXTREME CAUTION")
        logger.warning("This is a placeholder implementation. No real trades will be executed.")

        # Get parameters
        strategies = kwargs.get('strategies', None)
        if strategies is None:
            strategies = create_strategies()

        # Start live data stream
        logger.info("Starting live data stream")
        self.is_running = True
        self.data_handler.start_live_data_stream(self._on_live_data)

        # Keep the main thread alive
        while self.is_running and not self.stop_event.is_set():
            time.sleep(1)

        # Stop live data stream
        self.data_handler.stop_live_data_stream()

        return {
            'status': 'completed',
            'message': 'Live trading placeholder completed'
        }

    def _on_live_data(self, tick_data):
        """Process live tick data"""
        if not self.is_running or self.stop_event.is_set():
            return

        # Placeholder for live trading logic
        logger.info(f"Received tick: {tick_data['time']} - Price: {tick_data['close']}")

    def stop(self):
        """Stop live trading"""
        self.stop_event.set()
        logger.info("Live trading stop signal received")


class TradingBot:
    """Main trading bot class"""

    def __init__(self):
        self.data_handler = DataHandler()
        self.mode = None
        self.strategies = None

        # Create results directory if it doesn't exist
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    def load_data(self, source: str = "local_csv", path: str = None, start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
        """Load historical data"""
        return self.data_handler.load_historical_data(source, path, start_date, end_date)

    def set_mode(self, mode: str) -> bool:
        """Set trading mode"""
        mode_mapping = {
            'backtest': BacktestMode,
            'train': TrainingMode,
            'paper': PaperTradingMode,
            'live': LiveTradingMode
        }

        if mode not in mode_mapping:
            logger.error(f"Invalid mode: {mode}. Must be one of {list(mode_mapping.keys())}")
            return False

        self.mode = mode_mapping[mode]()
        logger.info(f"Set mode to {mode}")
        return True

    def load_strategies(self) -> Dict[str, TradingStrategy]:
        """Load trading strategies"""
        self.strategies = create_strategies()
        logger.info(f"Loaded {len(self.strategies)} strategies")
        return self.strategies

    def run(self, **kwargs) -> Dict:
        """Run trading bot with current mode"""
        if self.mode is None:
            logger.error("No mode set. Use set_mode() first.")
            return {'error': 'No mode set'}

        # Load data if needed
        data = kwargs.get('data', None)
        if data is None and kwargs.get('load_data', True):
            data = self.load_data()
            if len(data) == 0:
                logger.error("Failed to load data")
                return {'error': 'Failed to load data'}

        # Load strategies if needed
        if self.strategies is None and kwargs.get('load_strategies', True):
            self.load_strategies()

        # Add strategies to kwargs
        kwargs['strategies'] = kwargs.get('strategies', self.strategies)

        # Run the selected mode
        logger.info(f"Running trading bot in {self.mode.__class__.__name__}")
        result = self.mode.execute(data, **kwargs)

        return result

    def stop(self):
        """Stop the trading bot"""
        if hasattr(self.mode, 'stop') and callable(self.mode.stop):
            self.mode.stop()
            logger.info("Stopped trading bot")
            return True
        return False

    def generate_report(self, results: Union[Dict, BacktestResult], report_path: str = None) -> str:
        """Generate HTML report from results"""
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(config.RESULTS_DIR, f"report_{timestamp}.html")

        # Simple HTML report generation
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NQ E-mini Trading Bot Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 12px; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #ddd; }}
                .chart {{ margin: 20px 0; }}
                .footer {{ margin-top: 40px; color: #666; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>NQ E-mini Trading Bot Report</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """

        # Add appropriate content based on result type
        if isinstance(results, BacktestResult):
            html += f"""
                    <p>Strategy: {results.strategy_name}</p>
                    <p>Initial Balance: ${results.initial_balance:.2f}</p>
                    <p>Final Balance: ${results.equity_curve.iloc[-1]:.2f}</p>
                    <p>Total Return: ${results.equity_curve.iloc[-1] - results.initial_balance:.2f} ({(results.equity_curve.iloc[-1] / results.initial_balance - 1) * 100:.2f}%)</p>
                    <p>Total Trades: {results.metrics['total_trades']}</p>
                    <p>Win Rate: {results.metrics.get('win_rate', 0):.2f}%</p>
                    <p>Max Drawdown: {results.metrics.get('max_drawdown', 0):.2f}%</p>
                </div>

                <h2>Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """

            # Add metric rows
            for key, value in results.metrics.items():
                formatted_key = key.replace('_', ' ').title()

                if isinstance(value, float):
                    if 'ratio' in key or 'factor' in key:
                        formatted_value = f"{value:.2f}"
                    elif 'pnl' in key or key == 'avg_win' or key == 'avg_loss':
                        formatted_value = f"${value:.2f}"
                    elif 'rate' in key or 'roi' in key or 'drawdown' in key:
                        formatted_value = f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)

                html += f"""
                    <tr>
                        <td>{formatted_key}</td>
                        <td>{formatted_value}</td>
                    </tr>
                """

            html += """
                </table>

                <h2>Trade Analysis</h2>
            """

            if results.trades:
                html += """
                <table>
                    <tr>
                        <th>Entry Time</th>
                        <th>Entry Price</th>
                        <th>Direction</th>
                        <th>Exit Time</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                        <th>Duration</th>
                    </tr>
                """

                # Add up to 50 most recent trades
                for trade in results.trades[-50:]:
                    duration_str = f"{trade.duration.total_seconds() / 60:.1f} min" if trade.duration else "N/A"
                    direction = "Long" if trade.direction > 0 else "Short"

                    html += f"""
                    <tr>
                        <td>{trade.entry_time.strftime("%Y-%m-%d %H:%M")}</td>
                        <td>{trade.entry_price:.2f}</td>
                        <td>{direction}</td>
                        <td>{trade.exit_time.strftime("%Y-%m-%d %H:%M") if trade.exit_time else "Open"}</td>
                        <td>{trade.exit_price:.2f if trade.exit_price else "N/A"}</td>
                        <td>${trade.pnl:.2f if trade.pnl is not None else "N/A"}</td>
                        <td>{duration_str}</td>
                    </tr>
                    """

                html += """
                </table>
                """

            # Add charts
            equity_curve_path = os.path.join(config.RESULTS_DIR, "backtest_results",
                                             results.strategy_name.replace(" ", "_").lower(),
                                             f"{results.strategy_name.replace(' ', '_').lower()}_results.png")

            trades_chart_path = os.path.join(config.RESULTS_DIR, "backtest_results",
                                             results.strategy_name.replace(" ", "_").lower(),
                                             f"{results.strategy_name.replace(' ', '_').lower()}_trades.png")

            if os.path.exists(equity_curve_path):
                html += f"""
                <div class="chart">
                    <h2>Equity Curve</h2>
                    <img src="{os.path.relpath(equity_curve_path, os.path.dirname(report_path))}" alt="Equity Curve" style="width: 100%;">
                </div>
                """

            if os.path.exists(trades_chart_path):
                html += f"""
                <div class="chart">
                    <h2>Trades</h2>
                    <img src="{os.path.relpath(trades_chart_path, os.path.dirname(report_path))}" alt="Trades Chart" style="width: 100%;">
                </div>
                """

        elif isinstance(results, dict):
            # For other result types
            html += """
                </div>

                <h2>Results</h2>
                <table>
                    <tr>
                        <th>Key</th>
                        <th>Value</th>
                    </tr>
            """

            # Add result items, excluding large lists/dicts
            for key, value in results.items():
                if isinstance(value, (list, dict)) and len(str(value)) > 1000:
                    value_str = f"{type(value).__name__} with {len(value)} items"
                elif isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                html += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value_str}</td>
                    </tr>
                """

            html += """
                </table>
            """

        # Close the HTML
        html += """
                <div class="footer">
                    <p>Generated by NQ E-mini Trading Bot</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Write to file
        with open(report_path, 'w') as f:
            f.write(html)

        logger.info(f"Report saved to {report_path}")
        return report_path


# Main entry point
def main():
    parser = argparse.ArgumentParser(description='NQ E-mini Trading Bot')

    parser.add_argument('--mode', choices=['backtest', 'train', 'paper', 'live'], default='backtest',
                        help='Trading mode')
    parser.add_argument('--data-source', choices=['local_csv', 'polygon', 'yfinance'], default='local_csv',
                        help='Data source')
    parser.add_argument('--data-path', type=str, default=config.HISTORICAL_DATA_PATH,
                        help='Path to historical data CSV')
    parser.add_argument('--start-date', type=str, help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, help='Strategy name (default: ensemble)')
    parser.add_argument('--position-size', type=int, default=config.MAX_POSITION_SIZE,
                        help='Position size in contracts')
    parser.add_argument('--use-stop-loss', action='store_true', default=True,
                        help='Use stop loss')
    parser.add_argument('--stop-loss-pct', type=float, default=config.STOP_LOSS_PCT,
                        help='Stop loss percentage')
    parser.add_argument('--report', action='store_true',
                        help='Generate HTML report')

    # Additional parameters for specific modes
    parser.add_argument('--episodes', type=int, default=config.RL_EPISODES,
                        help='Number of episodes for RL training')

    args = parser.parse_args()

    # Create trading bot
    bot = TradingBot()

    # Load data
    data = bot.load_data(args.data_source, args.data_path, args.start_date, args.end_date)

    if len(data) == 0:
        logger.error("Failed to load data")
        return

    # Load strategies
    strategies = bot.load_strategies()

    # Select strategy if specified
    # Select strategy if specified
    if args.strategy:
        if args.strategy in strategies:
            selected_strategy = strategies[args.strategy]
        else:
            logger.error(f"Strategy '{args.strategy}' not found")
            return
    else:
        # Use ensemble strategy by default
        selected_strategy = strategies['ensemble']

    # Set mode
    if not bot.set_mode(args.mode):
        return

    # Run bot with selected mode
    kwargs = {
        'data': data,
        'strategies': selected_strategy,
        'position_size': args.position_size,
        'use_stop_loss': args.use_stop_loss,
        'stop_loss_pct': args.stop_loss_pct
    }

    if args.mode == 'train':
        kwargs['episodes'] = args.episodes

    results = bot.run(**kwargs)

    # Generate report if requested
    if args.report:
        report_path = bot.generate_report(results)
        logger.info(f"Report generated at {report_path}")

    return results


if __name__ == "__main__":
    main()