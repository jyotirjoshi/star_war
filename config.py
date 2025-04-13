import os
from dotenv import load_dotenv
from datetime import time, datetime, timezone

# Load environment variables
load_dotenv()

# General settings
PROJECT_NAME = "NQ_Elite_Trading_Bot"
VERSION = "1.0.0"
LOG_LEVEL = "INFO"
DATA_DIR = "./data"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"

# Create necessary directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Trading parameters
SYMBOL = "NQ"  # E-mini Nasdaq-100 Futures
TIMEFRAME = "5min"
TRADING_HOURS_START = time(9, 30)  # Market open (Eastern Time)
TRADING_HOURS_END = time(16, 0)    # Market close (Eastern Time)
TRADING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Training parameters
HISTORICAL_DATA_PATH = r"C:\Users\spars\Desktop\kise pata\elite_bot\NQ_emini_5min_complete_1y_2025-04-10.csv"
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Reinforcement Learning parameters
RL_EPISODES = 1000
RL_MEMORY_SIZE = 10000
RL_GAMMA = 0.99  # Discount factor
RL_EPSILON = 1.0  # Exploration rate
RL_EPSILON_MIN = 0.01
RL_EPSILON_DECAY = 0.995

# Risk management
MAX_POSITION_SIZE = 1  # Max number of contracts
MAX_DAILY_LOSS = 1000  # Maximum allowed daily loss in dollars
MAX_DRAWDOWN_PCT = 5.0  # Maximum allowed drawdown percentage
STOP_LOSS_PCT = 0.5    # Stop loss as percentage of entry price
TAKE_PROFIT_PCT = 1.0  # Take profit as percentage of entry price
TRAILING_STOP_PCT = 0.3  # Trailing stop percentage

# API keys and access (replace with your own in .env file)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_ENDPOINT = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")

# Data sources
DATA_SOURCES = {
    "historical": "local_csv",  # Options: local_csv, polygon, alpaca, yfinance
    "live": "polygon"           # Options: polygon, alpaca, interactive_brokers
}

# Strategy weights (for ensemble approach)
STRATEGY_WEIGHTS = {
    "trend_following": 0.15,
    "mean_reversion": 0.15,
    "breakout": 0.1,
    "vwap": 0.1,
    "support_resistance": 0.1,
    "momentum": 0.1,
    "volume_profile": 0.05,
    "bollinger_bands": 0.05,
    "macd": 0.05,
    "rsi": 0.05,
    "stochastic": 0.05,
    "ichimoku": 0.05,
    "order_flow": 0.05,
    "market_profile": 0.05,
    "deep_learning": 0.1,
    "rl_agent": 0.15
}

# Paper trading settings
PAPER_TRADING_BALANCE = 100000  # Starting balance for paper trading
COMMISSION_PER_CONTRACT = 4.50  # Average commission per contract (round trip)
SLIPPAGE_TICKS = 1  # Expected slippage in ticks

# Performance tracking
PERFORMANCE_METRICS = [
    "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
    "win_rate", "profit_factor", "avg_win", "avg_loss", "risk_reward_ratio"
]
# Alpaca API settings (add these to your existing config.py)
ALPHA_VANTAGE_KEY = "DPPDLIMWZ9GR8PS8"