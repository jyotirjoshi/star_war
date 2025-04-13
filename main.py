import os
import sys
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import config
from data_handler import DataHandler
from strategies import create_strategies
from rl_agent import RLTrainer
from backtester import Backtester
from trading_bot import TradingBot, BacktestMode, TrainingMode, PaperTradingMode, LiveTradingMode

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.RESULTS_DIR, "main.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")


def print_logo():
    """Print ASCII art logo"""
    logo = """
    ███╗   ██╗ ██████╗       ███████╗██╗     ██╗████████╗███████╗    ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗ 
    ████╗  ██║██╔═══██╗      ██╔════╝██║     ██║╚══██╔══╝██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
    ██╔██╗ ██║██║   ██║█████╗█████╗  ██║     ██║   ██║   █████╗         ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝
    ██║╚██╗██║██║▄▄ ██║╚════╝██╔══╝  ██║     ██║   ██║   ██╔══╝         ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗
    ██║ ╚████║╚██████╔╝      ███████╗███████╗██║   ██║   ███████╗       ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║
    ╚═╝  ╚═══╝ ╚══▀▀═╝       ╚══════╝╚══════╝╚═╝   ╚═╝   ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝

                                    NQ E-mini Futures Algorithmic Trading Bot
    """
    print(logo)


def run_data_explorer(data_path):
    """Run data explorer mode"""
    print("\n=== Data Explorer ===")

    # Load data
    data_handler = DataHandler()
    data = data_handler.load_historical_data(path=data_path)

    if len(data) == 0:
        print("Failed to load data")
        return

    # Show basic statistics
    stats = data_handler.get_data_stats(data)

    print("\nData Statistics:")
    print(f"Start Date: {stats['start_date']}")
    print(f"End Date: {stats['end_date']}")
    print(f"Trading Days: {stats['trading_days']}")
    print(f"Total Rows: {stats['total_rows']}")
    print(f"Average Daily Range: {stats['avg_daily_range']:.2f}")
    print(f"Average Daily Volume: {stats['avg_daily_volume']:.2f}")
    print(f"Price Standard Deviation: {stats['price_std']:.2f}")
    print(f"Average Returns: {stats['avg_returns']:.4f}%")
    print(f"Maximum Return: {stats['max_return']:.4f}%")
    print(f"Minimum Return: {stats['min_return']:.4f}%")

    # Show first and last few rows
    print("\nFirst 5 rows:")
    print(data.head())

    print("\nLast 5 rows:")
    print(data.tail())

    # Plot data
    plt.figure(figsize=(15, 10))

    # Price chart
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title('Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)

    # Volume chart
    plt.subplot(2, 1, 2)
    plt.bar(data.index, data['volume'], color='blue', alpha=0.6)
    plt.title('Volume Chart')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True)

    plt.tight_layout()

    # Save chart
    chart_path = os.path.join(config.RESULTS_DIR, "data_explorer_chart.png")
    plt.savefig(chart_path)

    print(f"\nChart saved to {chart_path}")
    print("You can open this chart to visualize your data")


def run_strategy_tester(data_path, strategy_name=None):
    """Run strategy tester mode"""
    print("\n=== Strategy Tester ===")

    # Load data
    data_handler = DataHandler()
    data = data_handler.load_historical_data(path=data_path)

    if len(data) == 0:
        print("Failed to load data")
        return

    # Load strategies
    strategies = create_strategies()

    if strategy_name:
        if strategy_name in strategies:
            selected_strategies = {strategy_name: strategies[strategy_name]}
        else:
            print(f"Strategy '{strategy_name}' not found")
            return
    else:
        # Test all strategies
        selected_strategies = strategies

    # Create backtester
    backtester = Backtester(data)

    # Run backtest for each strategy
    results = {}

    for name, strategy in selected_strategies.items():
        print(f"\nTesting {name} strategy...")
        result = backtester.run_backtest(strategy)
        results[name] = result

        # Print summary
        result.print_summary()

        # Save results
        result_folder = os.path.join(config.RESULTS_DIR, "backtest_results", name.replace(" ", "_").lower())
        result.save_results(result_folder)

    # Compare strategies
    if len(results) > 1:
        print("\nComparing strategies...")
        backtester.compare_strategies(results)

        # Plot comparison
        comparison_path = os.path.join(config.RESULTS_DIR, "strategy_comparison.png")
        backtester.plot_comparison(results, comparison_path)
        print(f"Comparison chart saved to {comparison_path}")


def run_reinforcement_learning(data_path, episodes=None):
    """Run reinforcement learning mode"""
    print("\n=== Reinforcement Learning ===")

    # Load data
    data_handler = DataHandler()
    data = data_handler.load_historical_data(path=data_path)

    if len(data) == 0:
        print("Failed to load data")
        return

    # Set episodes
    if episodes is None:
        episodes = config.RL_EPISODES

    # Create trainer
    trainer = RLTrainer(data)

    # Train agent
    print(f"Training RL agent for {episodes} episodes...")
    training_results = trainer.train(episodes=episodes)

    # Evaluate agent
    print("\nEvaluating RL agent...")
    best_model_path = os.path.join(config.MODELS_DIR, "dqn_trading_best.h5")
    eval_results = trainer.evaluate(best_model_path)

    # Plot results
    trainer.plot_results(training_results, eval_results)

    # Print evaluation metrics
    print("\nEvaluation Results:")
    print(f"Initial Balance: ${eval_results['initial_balance']:.2f}")
    print(f"Final Balance: ${eval_results['final_balance']:.2f}")
    print(f"Total Return: ${eval_results['total_return']:.2f} ({eval_results['roi']:.2f}%)")
    print(f"Sharpe Ratio: {eval_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {eval_results['max_drawdown']:.2f}%")
    print(f"Total Trades: {eval_results['total_trades']}")
    print(f"Win Rate: {eval_results['win_rate']:.2f}%")

    # Return results
    return {
        'training_results': training_results,
        'eval_results': eval_results,
        'model_path': best_model_path
    }


def run_paper_trading(data_path, strategy_name=None, duration=None):
    """Run paper trading mode"""
    print("\n=== Paper Trading ===")

    # Load data
    data_handler = DataHandler()
    data = data_handler.load_historical_data(path=data_path)

    if len(data) == 0:
        print("Failed to load data")
        return

    # Load strategies
    strategies = create_strategies()

    if strategy_name:
        if strategy_name in strategies:
            selected_strategy = strategies[strategy_name]
        else:
            print(f"Strategy '{strategy_name}' not found")
            return
    else:
        # Use ensemble strategy by default
        selected_strategy = strategies['ensemble']

    # Create trading bot
    bot = TradingBot()
    bot.set_mode('paper')

    # If duration is specified, use only that portion of data
    if duration:
        try:
            days = int(duration)
            data = data.iloc[-days * 78:]  # Assuming 78 5-min bars per day
        except:
            print(f"Invalid duration: {duration}")

    # Run paper trading
    print(f"Starting paper trading simulation with {selected_strategy.name} strategy...")
    results = bot.run(
        data=data,
        strategies=selected_strategy,
        position_size=config.MAX_POSITION_SIZE,
        use_stop_loss=True,
        stop_loss_pct=config.STOP_LOSS_PCT
    )

    # Print results
    print("\nPaper Trading Results:")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    profit = results['final_balance'] - results['initial_balance']
    print(f"Profit/Loss: ${profit:.2f} ({profit / results['initial_balance'] * 100:.2f}%)")
    print(f"Total Trades: {results['total_trades']}")

    # Print trade details
    if results['trades']:
        print("\nTrade Details:")
        for i, trade in enumerate(results['trades'][-10:], 1):  # Show last 10 trades
            print(
                f"{i}. {trade['entry_time']} - {trade['exit_time']}: {trade['direction']} @ {trade['entry_price']:.2f} -> {trade['exit_price']:.2f}, PnL: ${trade['pnl']:.2f}")

    return results


def main():
    """Main entry point"""
    # Create argument parser
    parser = argparse.ArgumentParser(description='NQ E-mini Elite Trading Bot')

    parser.add_argument('mode', choices=['explore', 'backtest', 'optimize', 'train', 'paper', 'live'],
                        help='Operating mode')
    parser.add_argument('--data-path', type=str, default=config.HISTORICAL_DATA_PATH,
                        help='Path to historical data CSV')
    parser.add_argument('--strategy', type=str, help='Strategy name')
    parser.add_argument('--episodes', type=int, help='Number of episodes for RL training')
    parser.add_argument('--duration', type=str, help='Duration for paper trading (in days)')

    args = parser.parse_args()

    # Print logo
    print_logo()

    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        print(f"Error: Data file not found: {args.data_path}")
        return

    try:
        # Run selected mode
        if args.mode == 'explore':
            run_data_explorer(args.data_path)
        elif args.mode == 'backtest':
            run_strategy_tester(args.data_path, args.strategy)
        elif args.mode == 'train':
            run_reinforcement_learning(args.data_path, args.episodes)
        elif args.mode == 'paper':
            run_paper_trading(args.data_path, args.strategy, args.duration)
        elif args.mode == 'live':
            print("\n=== Live Trading Mode ===")
            print("WARNING: Live trading mode is not enabled in this version.")
            print("To enable live trading, you need to:")
            print("1. Configure a proper broker API connection in config.py")
            print("2. Extensively test your strategies in paper trading mode")
            print("3. Understand the risks involved in algorithmic trading")
        elif args.mode == 'optimize':
            print("\n=== Strategy Optimization Mode ===")
            print("Strategy optimization is available in the premium version.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.exception("Error in main function")
        print(f"\nError: {e}")

    print("\nThank you for using NQ E-mini Elite Trading Bot!")


if __name__ == "__main__":
    main()