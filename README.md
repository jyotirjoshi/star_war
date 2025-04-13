# NQ E-mini Elite Trading Bot

A sophisticated algorithmic trading bot for day trading E-mini Nasdaq-100 futures (NQ) featuring multiple strategies, reinforcement learning, and extensive backtesting capabilities.

## Features

- **Multiple Trading Strategies**: Includes 15+ advanced strategies from trend-following to machine learning-based approaches
- **Reinforcement Learning**: Self-improving agent that learns optimal trading decisions
- **Comprehensive Backtesting**: Test strategies on historical data with detailed performance metrics
- **Paper Trading**: Simulate real trading without risking real money
- **Data Acquisition**: Handles both historical and live market data
- **Advanced Analytics**: Detailed performance metrics and visualization tools

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nq-elite-trading-bot.git
cd nq-elite-trading-bot
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Install TA-Lib (may require separate installation steps depending on your OS):
   - Windows: Download and install from [TA-Lib website](https://ta-lib.org/)
   - Linux: `sudo apt-get install ta-lib`
   - macOS: `brew install ta-lib`

4. Create a `.env` file with your API keys (if using external data services):
```
POLYGON_API_KEY=your_polygon_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
```

## Directory Structure

```
nq-elite-trading-bot/
├── config.py                # Configuration settings
├── data_handler.py          # Data acquisition and processing
├── strategies.py            # Trading strategies implementation
├── rl_agent.py              # Reinforcement learning agent
├── backtester.py            # Backtesting engine
├── trading_bot.py           # Main trading bot class
├── main.py                  # Entry point script
├── data/                    # Data storage
├── models/                  # Saved ML models
└── results/                 # Backtest results and reports
```

## Usage

### Data Exploration

Explore and visualize your data:

```bash
python main.py explore --data-path path/to/your/data.csv
```

### Backtesting

Test one or all strategies on historical data:

```bash
python main.py backtest --data-path path/to/your/data.csv --strategy strategy_name
```

To test all strategies, omit the `--strategy` parameter.

### Training the RL Agent

Train the reinforcement learning agent:

```bash
python main.py train --data-path path/to/your/data.csv --episodes 1000
```

### Paper Trading

Run paper trading simulation:

```bash
python main.py paper --data-path path/to/your/data.csv --strategy ensemble --duration 30
```

This will run a simulation over 30 days of data with the ensemble strategy.

### Live Trading (disabled by default)

For live trading, you'll need to:
1. Configure your broker API connection in `config.py`
2. Thoroughly test your strategies in backtesting and paper trading modes
3. Understand the risks involved

## Trading Strategies

The bot includes the following strategies:

1. **Trend Following Strategy**: Moving average crossovers with ADX confirmation
2. **Mean Reversion Strategy**: Bollinger Bands with RSI filters
3. **Breakout Strategy**: Price level breakouts with volume confirmation
4. **VWAP Strategy**: Trades based on price relation to VWAP
5. **Support Resistance Strategy**: Identifies and trades key S/R levels
6. **Momentum Strategy**: Multiple momentum indicators for trend strength
7. **Volume Profile Strategy**: Trades based on volume distribution
8. **Bollinger Bands Strategy**: Enhanced Bollinger Band strategy with squeeze detection
9. **MACD Strategy**: MACD with divergence detection
10. **RSI Strategy**: RSI with centerline and divergence signals
11. **Stochastic Strategy**: Stochastic oscillator with overbought/oversold filters
12. **Ichimoku Strategy**: Complete Ichimoku Cloud system
13. **Order Flow Strategy**: Simulated order flow analysis
14. **Market Profile Strategy**: Value area trading based on market profile
15. **Deep Learning Strategy**: Neural network price prediction
16. **Ensemble Strategy**: Weighted combination of all strategies

## Risk Management

The bot includes several risk management features:
- Maximum position size limits
- Stop loss protection
- Maximum daily loss limits
- Trailing stops
- Drawdown control

## Warning

Trading futures involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. This software is for educational and research purposes only.

## License

This project is licensed under the MIT License - see the LICENSE file for details.