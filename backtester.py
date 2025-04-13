import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backtester")


class Trade:
    """Class to represent a trade"""

    def __init__(self, entry_time, entry_price, direction, size=1, entry_signal=None):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time = None
        self.exit_price = None
        self.direction = direction  # 1 for long, -1 for short
        self.size = size
        self.pnl = None
        self.pnl_pct = None
        self.duration = None
        self.entry_signal = entry_signal
        self.exit_signal = None

    def close_trade(self, exit_time, exit_price, exit_signal=None):
        """Close the trade and calculate P&L"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_signal = exit_signal

        # Calculate P&L
        price_diff = (exit_price - self.entry_price) * self.direction
        self.pnl = price_diff * self.size * 20  # 20 is the multiplier for NQ
        self.pnl_pct = (price_diff / self.entry_price) * 100

        # Calculate duration
        self.duration = exit_time - self.entry_time

    def to_dict(self):
        """Convert trade to dictionary"""
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'direction': self.direction,
            'size': self.size,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'duration_minutes': self.duration.total_seconds() / 60 if self.duration else None,
            'entry_signal': self.entry_signal,
            'exit_signal': self.exit_signal
        }


class BacktestResult:
    """Class to store backtest results"""

    def __init__(self, initial_balance, trades, data, strategy_name):
        self.initial_balance = initial_balance
        self.trades = trades
        self.data = data
        self.strategy_name = strategy_name
        self.metrics = {}
        self.equity_curve = None
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate performance metrics"""
        # Initialize metrics
        self.metrics = {
            'initial_balance': self.initial_balance,
            'total_trades': len(self.trades),
            'win_rate': 0,
            'profit_factor': 0,
            'total_pnl': 0,
            'avg_trade_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0
        }

        # Calculate equity curve
        equity = pd.Series(self.initial_balance, index=[self.data.index[0]])

        # Add completed trades to equity curve
        for trade in self.trades:
            if trade.exit_time is not None and trade.pnl is not None:
                equity[trade.exit_time] = equity.iloc[-1] + trade.pnl

        # Reindex to fill in all dates
        self.equity_curve = equity.reindex(self.data.index, method='ffill')

        # Calculate metrics if there are trades
        if self.trades:
            # Count wins and losses
            wins = sum(1 for t in self.trades if t.pnl is not None and t.pnl > 0)
            losses = sum(1 for t in self.trades if t.pnl is not None and t.pnl < 0)

            # Win rate
            self.metrics['win_rate'] = (wins / len(self.trades)) * 100 if len(self.trades) > 0 else 0

            # Profit factor
            total_profit = sum(t.pnl for t in self.trades if t.pnl is not None and t.pnl > 0)
            total_loss = sum(abs(t.pnl) for t in self.trades if t.pnl is not None and t.pnl < 0)
            self.metrics[
                'profit_factor'] = total_profit / total_loss if total_loss > 0 else 0 if total_profit == 0 else float(
                'inf')

            # Total P&L and ROI
            self.metrics['total_pnl'] = sum(t.pnl for t in self.trades if t.pnl is not None)
            self.metrics['roi'] = (self.metrics['total_pnl'] / self.initial_balance) * 100

            # Average trade metrics
            self.metrics['avg_trade_pnl'] = self.metrics['total_pnl'] / len(self.trades)
            self.metrics['avg_win'] = total_profit / wins if wins > 0 else 0
            self.metrics['avg_loss'] = -total_loss / losses if losses > 0 else 0

            # Trade duration
            durations = [t.duration.total_seconds() / 60 for t in self.trades if t.duration is not None]
            if durations:
                self.metrics['avg_trade_duration_min'] = sum(durations) / len(durations)

            # Max drawdown
            peak = self.equity_curve.expanding().max()
            drawdown = ((self.equity_curve - peak) / peak) * 100
            self.metrics['max_drawdown'] = abs(drawdown.min())

            # Sharpe and Sortino ratios
            daily_returns = self.equity_curve.pct_change().dropna()

            if len(daily_returns) > 1:
                sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
                self.metrics['sharpe_ratio'] = sharpe

                downside_returns = daily_returns[daily_returns < 0]
                sortino = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(
                    downside_returns) > 0 and downside_returns.std() != 0 else 0
                self.metrics['sortino_ratio'] = sortino
            else:
                self.metrics['sharpe_ratio'] = 0
                self.metrics['sortino_ratio'] = 0

    def plot_results(self, save_path=None):
        """Plot equity curve and drawdown"""
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.grid(True)

        # Drawdown
        plt.subplot(2, 1, 2)
        peak = self.equity_curve.expanding().max()
        drawdown = ((self.equity_curve - peak) / peak) * 100
        plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        plt.title('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def print_summary(self):
        """Print backtest summary"""
        print(f"\n===== Backtest Results: {self.strategy_name} =====")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.equity_curve.iloc[-1]:.2f}")
        print(f"Total Return: {self.metrics['roi']:.2f}%")
        print(f"Total Trades: {self.metrics['total_trades']}")
        print(f"Win Rate: {self.metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"Average Trade P&L: ${self.metrics['avg_trade_pnl']:.2f}")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print("=" * 40)


class Backtester:
    """Backtesting engine for trading strategies"""

    def __init__(self, data, initial_balance=100000):
        self.data = data
        self.initial_balance = initial_balance

    def run_backtest(self, strategy, **kwargs):
        """Run backtest for a strategy"""
        logger.info(f"Running backtest for {strategy}")

        # This is a placeholder implementation
        # In a real implementation, you would generate signals from the strategy
        # and execute trades based on those signals

        # Simulate some random trades
        trades = []
        for i in range(10):
            entry_time = self.data.index[int(len(self.data) * i / 10)]
            entry_price = self.data.loc[entry_time, 'close']
            direction = 1 if i % 2 == 0 else -1

            trade = Trade(entry_time, entry_price, direction)

            # Exit after 5 bars
            exit_idx = min(int(len(self.data) * (i + 1) / 10) - 1, len(self.data) - 1)
            exit_time = self.data.index[exit_idx]
            exit_price = self.data.loc[exit_time, 'close']

            trade.close_trade(exit_time, exit_price)
            trades.append(trade)

        # Create and return backtest result
        return BacktestResult(self.initial_balance, trades, self.data, str(strategy))

    def compare_strategies(self, results):
        """Compare multiple backtest results"""
        if not results:
            logger.warning("No results to compare")
            return None

        # Create comparison table
        comparison = {}
        for name, result in results.items():
            comparison[name] = {
                'total_return': result.metrics['roi'],
                'win_rate': result.metrics['win_rate'],
                'profit_factor': result.metrics['profit_factor'],
                'max_drawdown': result.metrics['max_drawdown'],
                'sharpe_ratio': result.metrics['sharpe_ratio']
            }

        return pd.DataFrame(comparison)

    def plot_comparison(self, results, save_path=None):
        """Plot equity curves for multiple strategies"""
        plt.figure(figsize=(12, 8))

        for name, result in results.items():
            plt.plot(result.equity_curve, label=name)

        plt.title('Strategy Comparison')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2025-01-01', periods=100)
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100)
    }, index=dates)

    # Create backtester
    backtester = Backtester(data)

    # Run backtest
    result = backtester.run_backtest("Test Strategy")

    # Print results
    result.print_summary()
    result.plot_results()