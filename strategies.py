import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
import logging
import os
import talib
from datetime import datetime
import math
import config
from scipy import stats

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.RESULTS_DIR, "strategies.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Strategies")


class TradingStrategy:
    """Base class for all trading strategies"""

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {name} strategy")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals: 1 (buy), -1 (sell), 0 (hold)"""
        raise NotImplementedError("Each strategy must implement generate_signals")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific features"""
        raise NotImplementedError("Each strategy must implement calculate_features")


class TrendFollowingStrategy(TradingStrategy):
    """Simple moving average crossover strategy"""

    def __init__(self, fast_period: int = 10, slow_period: int = 50):
        super().__init__("Trend Following")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()

        # Calculate additional trend indicators
        df['adx'] = talib.ADX(df['high'].values, df['low'].values,
                              df['close'].values, timeperiod=14)

        # Trend strength
        df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)

        # Directional indicators
        df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values,
                                      df['close'].values, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values,
                                        df['close'].values, timeperiod=14)

        # Trend direction
        df['trend_direction'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Generate signals based on moving average crossover
        signals = np.where((df['fast_ma'] > df['slow_ma']) &
                           (df['trend_strength'] == 1) &
                           (df['trend_direction'] == 1), 1,
                           np.where((df['fast_ma'] < df['slow_ma']) &
                                    (df['trend_strength'] == 1) &
                                    (df['trend_direction'] == -1), -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy based on Bollinger Bands"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Mean Reversion")
        self.period = period
        self.std_dev = std_dev

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=self.period).mean()
        std = df['close'].rolling(window=self.period).std()
        df['upper_band'] = df['middle_band'] + (std * self.std_dev)
        df['lower_band'] = df['middle_band'] - (std * self.std_dev)

        # Calculate %B (Percent Bandwidth)
        df['percent_b'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

        # Calculate Bandwidth
        df['bandwidth'] = (df['upper_band'] - df['lower_band']) / df['middle_band']

        # RSI for confirmation
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Generate signals based on price relative to Bollinger Bands
        signals = np.where((df['close'] < df['lower_band']) & (df['rsi'] < 30), 1,
                           np.where((df['close'] > df['upper_band']) & (df['rsi'] > 70), -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class BreakoutStrategy(TradingStrategy):
    """Breakout strategy based on price levels and volume confirmation"""

    def __init__(self, lookback_period: int = 20):
        super().__init__("Breakout")
        self.lookback_period = lookback_period

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate recent highs and lows
        df['recent_high'] = df['high'].rolling(window=self.lookback_period).max()
        df['recent_low'] = df['low'].rolling(window=self.lookback_period).min()

        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=self.lookback_period).mean()
        df['volume_ratio'] = df['volume'] / df['avg_volume']

        # ATR for volatility measurement
        df['atr'] = talib.ATR(df['high'].values, df['low'].values,
                              df['close'].values, timeperiod=14)

        # Calculate price distance from recent highs/lows
        df['dist_from_high'] = (df['recent_high'] - df['close']) / df['atr']
        df['dist_from_low'] = (df['close'] - df['recent_low']) / df['atr']

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Generate signals based on breakouts with volume confirmation
        signals = np.where((df['close'] > df['recent_high'].shift(1)) &
                           (df['volume_ratio'] > 1.5), 1,
                           np.where((df['close'] < df['recent_low'].shift(1)) &
                                    (df['volume_ratio'] > 1.5), -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class VWAPStrategy(TradingStrategy):
    """VWAP (Volume Weighted Average Price) based strategy"""

    def __init__(self):
        super().__init__("VWAP")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Add a date column for grouping
        df['date'] = df.index.date

        # Calculate cumulative values for VWAP
        df['vwap'] = np.nan

        # Group by date to calculate daily VWAP
        for date, group in df.groupby('date'):
            cumulative_pv = (group['close'] * group['volume']).cumsum()
            cumulative_volume = group['volume'].cumsum()
            group_vwap = cumulative_pv / cumulative_volume

            df.loc[group.index, 'vwap'] = group_vwap

        # Calculate standard deviations from VWAP
        df['vwap_std'] = df.groupby('date')['close'].transform(
            lambda x: (x - df.loc[x.index, 'vwap']) / x.std()
        )

        # Calculate VWAP bands
        df['vwap_upper1'] = df['vwap'] * 1.01  # 1% upper band
        df['vwap_lower1'] = df['vwap'] * 0.99  # 1% lower band
        df['vwap_upper2'] = df['vwap'] * 1.02  # 2% upper band
        df['vwap_lower2'] = df['vwap'] * 0.98  # 2% lower band

        # Drop the date column as it's no longer needed
        df = df.drop('date', axis=1)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Generate signals based on price relative to VWAP
        signals = np.where((df['close'] < df['vwap_lower2']) &
                           (df['close'].shift(1) > df['vwap_lower2'].shift(1)), 1,
                           np.where((df['close'] > df['vwap_upper2']) &
                                    (df['close'].shift(1) < df['vwap_upper2'].shift(1)), -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class SupportResistanceStrategy(TradingStrategy):
    """Support and Resistance levels strategy"""

    def __init__(self, num_levels: int = 3, lookback_period: int = 50):
        super().__init__("Support Resistance")
        self.num_levels = num_levels
        self.lookback_period = lookback_period

    def find_pivot_points(self, data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Find pivot points in the data"""
        highs = data['high'].values
        lows = data['low'].values

        pivot_highs = np.zeros_like(highs)
        pivot_lows = np.zeros_like(lows)

        for i in range(n, len(highs) - n):
            # Check for pivot high
            if all(highs[i] > highs[i - j] for j in range(1, n + 1)) and \
                    all(highs[i] > highs[i + j] for j in range(1, n + 1)):
                pivot_highs[i] = highs[i]

            # Check for pivot low
            if all(lows[i] < lows[i - j] for j in range(1, n + 1)) and \
                    all(lows[i] < lows[i + j] for j in range(1, n + 1)):
                pivot_lows[i] = lows[i]

        result = pd.DataFrame({
            'pivot_high': pivot_highs,
            'pivot_low': pivot_lows
        }, index=data.index)

        result['pivot_high'] = result['pivot_high'].replace(0, np.nan)
        result['pivot_low'] = result['pivot_low'].replace(0, np.nan)

        return result

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Find pivot points
        pivots = self.find_pivot_points(df)
        df['pivot_high'] = pivots['pivot_high']
        df['pivot_low'] = pivots['pivot_low']

        # Forward fill to create support and resistance levels
        df['support'] = df['pivot_low'].ffill()
        df['resistance'] = df['pivot_high'].ffill()

        # Calculate distance from current levels
        df['dist_to_support'] = (df['close'] - df['support']) / df['close'] * 100
        df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close'] * 100

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Generate signals based on price approaching support/resistance
        signals = np.where((df['dist_to_support'] < 0.3) & (df['dist_to_support'] > 0), 1,
                           np.where((df['dist_to_resistance'] < 0.3) & (df['dist_to_resistance'] > 0), -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class MomentumStrategy(TradingStrategy):
    """Momentum-based trading strategy using multiple indicators"""

    def __init__(self):
        super().__init__("Momentum")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate momentum indicators
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'].values, df['low'].values, df['close'].values,
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
            df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Calculate OBV (On-Balance Volume)
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        df['obv_ma'] = df['obv'].rolling(window=20).mean()

        # Calculate rate of change
        df['roc'] = talib.ROC(df['close'].values, timeperiod=10)

        # Calculate money flow index
        df['mfi'] = talib.MFI(
            df['high'].values, df['low'].values,
            df['close'].values, df['volume'].values, timeperiod=14
        )

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Generate signals based on momentum indicators
        buy_signals = (
                (df['rsi'] < 40) & (df['rsi'].shift(1) < 30) & (df['rsi'] > df['rsi'].shift(1)) &
                (df['macd'] > df['macdsignal']) &
                (df['obv'] > df['obv_ma']) &
                (df['roc'] > 0)
        )

        sell_signals = (
                (df['rsi'] > 60) & (df['rsi'].shift(1) > 70) & (df['rsi'] < df['rsi'].shift(1)) &
                (df['macd'] < df['macdsignal']) &
                (df['obv'] < df['obv_ma']) &
                (df['roc'] < 0)
        )

        signals = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class VolumeProfileStrategy(TradingStrategy):
    """Volume Profile strategy utilizing price by volume analysis"""

    def __init__(self, num_bins: int = 20, lookback_period: int = 30):
        super().__init__("Volume Profile")
        self.num_bins = num_bins
        self.lookback_period = lookback_period

    def calculate_volume_profile(self, prices: np.array, volumes: np.array, num_bins: int) -> Tuple[np.array, np.array]:
        """Calculate volume profile for a given price range"""
        if len(prices) == 0:
            return np.array([]), np.array([])

        min_price = np.min(prices)
        max_price = np.max(prices)

        if max_price == min_price:
            return np.array([min_price]), np.array([np.sum(volumes)])

        # Create price bins
        bins = np.linspace(min_price, max_price, num_bins + 1)

        # Assign volumes to price bins
        bin_indices = np.digitize(prices, bins) - 1

        # Sum volumes for each bin
        bin_volumes = np.zeros(num_bins)
        for i in range(len(prices)):
            if 0 <= bin_indices[i] < num_bins:
                bin_volumes[bin_indices[i]] += volumes[i]

        # Calculate bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        return bin_centers, bin_volumes

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate rolling window volume profile
        df['vp_level'] = np.nan
        df['vp_volume'] = np.nan
        df['poc'] = np.nan  # Point of Control
        df['val_high'] = np.nan  # Value Area High
        df['val_low'] = np.nan  # Value Area Low

        # Group by day to calculate daily volume profile
        df['date'] = df.index.date

        for date, group in df.groupby('date'):
            if len(group) < 5:
                continue

            prices = (group['high'].values + group['low'].values) / 2
            volumes = group['volume'].values

            bin_centers, bin_volumes = self.calculate_volume_profile(prices, volumes, self.num_bins)

            if len(bin_centers) > 0:
                # Find Point of Control (price level with highest volume)
                poc_idx = np.argmax(bin_volumes)
                poc = bin_centers[poc_idx]

                # Calculate Value Area (70% of total volume)
                total_volume = np.sum(bin_volumes)
                target_volume = 0.7 * total_volume

                sorted_idx = np.argsort(bin_volumes)[::-1]  # Sort by volume (descending)

                cum_volume = 0
                included_idx = []

                for idx in sorted_idx:
                    included_idx.append(idx)
                    cum_volume += bin_volumes[idx]
                    if cum_volume >= target_volume:
                        break

                if included_idx:
                    val_high = bin_centers[max(included_idx)]
                    val_low = bin_centers[min(included_idx)]

                    # Assign values to the group
                    df.loc[group.index, 'poc'] = poc
                    df.loc[group.index, 'val_high'] = val_high
                    df.loc[group.index, 'val_low'] = val_low

        # Drop the date column
        df = df.drop('date', axis=1)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Generate signals based on price relative to POC and Value Area
        buy_signals = (
                (df['close'] < df['val_low']) &
                (df['close'].shift(1) >= df['val_low'].shift(1))
        )

        sell_signals = (
                (df['close'] > df['val_high']) &
                (df['close'].shift(1) <= df['val_high'].shift(1))
        )

        signals = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class BollingerBandStrategy(TradingStrategy):
    """Bollinger Bands strategy with enhancements"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Bands")
        self.period = period
        self.std_dev = std_dev

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=self.period).mean()
        df['std_dev'] = df['close'].rolling(window=self.period).std()
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * self.std_dev)
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * self.std_dev)

        # Calculate %B (Percent Bandwidth)
        df['percent_b'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

        # Calculate Bandwidth
        df['bandwidth'] = (df['upper_band'] - df['lower_band']) / df['middle_band']

        # Calculate bandwidth rate of change (expansion/contraction)
        df['bandwidth_roc'] = df['bandwidth'].pct_change(5)

        # Bollinger Band squeeze indicators (low volatility)
        df['squeeze'] = np.where(df['bandwidth'] < df['bandwidth'].rolling(window=50).quantile(0.2), 1, 0)

        # Trend indicators based on middle band
        df['trend'] = np.where(df['middle_band'] > df['middle_band'].shift(5), 1,
                               np.where(df['middle_band'] < df['middle_band'].shift(5), -1, 0))

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Generate signals based on Bollinger Band patterns

        # Bollinger Band bounce (mean reversion)
        bounce_buy = (
                (df['close'] < df['lower_band']) &
                (df['close'].shift(1) < df['lower_band'].shift(1)) &
                (df['close'] > df['close'].shift(1))
        )

        bounce_sell = (
                (df['close'] > df['upper_band']) &
                (df['close'].shift(1) > df['upper_band'].shift(1)) &
                (df['close'] < df['close'].shift(1))
        )

        # Bollinger Band breakout (trend following)
        breakout_buy = (
                (df['squeeze'].shift(1) == 1) &
                (df['squeeze'] == 0) &
                (df['close'] > df['upper_band']) &
                (df['bandwidth_roc'] > 0.05)
        )

        breakout_sell = (
                (df['squeeze'].shift(1) == 1) &
                (df['squeeze'] == 0) &
                (df['close'] < df['lower_band']) &
                (df['bandwidth_roc'] > 0.05)
        )

        # Combine signals with priority to breakouts
        signals = np.where(breakout_buy, 1,
                           np.where(breakout_sell, -1,
                                    np.where(bounce_buy, 1,
                                             np.where(bounce_sell, -1, 0))))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class MACDStrategy(TradingStrategy):
    """Enhanced MACD strategy"""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate MACD
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
            df['close'].values,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )

        # Calculate MACD histogram slope
        df['macdhist_slope'] = df['macdhist'] - df['macdhist'].shift(1)

        # Divergence detection
        df['close_roc'] = df['close'].pct_change(5)
        df['macd_roc'] = df['macd'].diff(5)

        # Potential divergence when price and MACD move in opposite directions
        df['bullish_divergence'] = np.where(
            (df['close'] < df['close'].shift(5)) &
            (df['macd'] > df['macd'].shift(5)), 1, 0
        )

        df['bearish_divergence'] = np.where(
            (df['close'] > df['close'].shift(5)) &
            (df['macd'] < df['macd'].shift(5)), 1, 0
        )

        # Zero line crossover
        df['zero_cross_up'] = np.where(
            (df['macd'] > 0) & (df['macd'].shift(1) < 0), 1, 0
        )

        df['zero_cross_down'] = np.where(
            (df['macd'] < 0) & (df['macd'].shift(1) > 0), 1, 0
        )

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Signal line crossover
        crossover_buy = (
                (df['macd'] > df['macdsignal']) &
                (df['macd'].shift(1) <= df['macdsignal'].shift(1))
        )

        crossover_sell = (
                (df['macd'] < df['macdsignal']) &
                (df['macd'].shift(1) >= df['macdsignal'].shift(1))
        )

        # Zero line crossover with confirmation
        zero_buy = (
                df['zero_cross_up'] == 1 &
                (df['macdhist'] > 0) &
                (df['macdhist_slope'] > 0)
        )

        zero_sell = (
                df['zero_cross_down'] == 1 &
                (df['macdhist'] < 0) &
                (df['macdhist_slope'] < 0)
        )

        # Divergence signals
        div_buy = (
                (df['bullish_divergence'] == 1) &
                (df['macd'] < 0) &  # Below zero for stronger signal
                (df['macdhist_slope'] > 0)  # Histogram increasing
        )

        div_sell = (
                (df['bearish_divergence'] == 1) &
                (df['macd'] > 0) &  # Above zero for stronger signal
                (df['macdhist_slope'] < 0)  # Histogram decreasing
        )

        # Combine signals with priority
        signals = np.where(div_buy | zero_buy | crossover_buy, 1,
                           np.where(div_sell | zero_sell | crossover_sell, -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class RSIStrategy(TradingStrategy):
    """Enhanced RSI strategy"""

    def __init__(self, period: int = 14):
        super().__init__("RSI")
        self.period = period

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.period)

        # Calculate RSI moving average for trend detection
        df['rsi_ma'] = df['rsi'].rolling(window=10).mean()

        # RSI divergence detection
        df['close_higher_high'] = (
                (df['close'] > df['close'].shift(1)) &
                (df['close'].shift(1) > df['close'].shift(2))
        )

        df['close_lower_low'] = (
                (df['close'] < df['close'].shift(1)) &
                (df['close'].shift(1) < df['close'].shift(2))
        )

        df['rsi_higher_high'] = (
                (df['rsi'] > df['rsi'].shift(1)) &
                (df['rsi'].shift(1) > df['rsi'].shift(2))
        )

        df['rsi_lower_low'] = (
                (df['rsi'] < df['rsi'].shift(1)) &
                (df['rsi'].shift(1) < df['rsi'].shift(2))
        )

        # Bullish and bearish divergences
        df['bullish_divergence'] = (
                df['close_lower_low'] & ~df['rsi_lower_low']
        )

        df['bearish_divergence'] = (
                df['close_higher_high'] & ~df['rsi_higher_high']
        )

        # RSI trend
        df['rsi_uptrend'] = df['rsi'] > df['rsi_ma']
        df['rsi_downtrend'] = df['rsi'] < df['rsi_ma']

        # RSI conditions
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_overbought'] = df['rsi'] > 70

        # RSI centerline crossover
        df['rsi_cross_up_50'] = (
                (df['rsi'] > 50) & (df['rsi'].shift(1) < 50)
        )

        df['rsi_cross_down_50'] = (
                (df['rsi'] < 50) & (df['rsi'].shift(1) > 50)
        )

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Oversold/Overbought with trend confirmation
        oversold_buy = (
                df['rsi_oversold'] &
                (df['rsi'] > df['rsi'].shift(1)) &  # RSI turning up
                (df['rsi'].shift(1) < df['rsi'].shift(2))  # Confirmed turn
        )

        overbought_sell = (
                df['rsi_overbought'] &
                (df['rsi'] < df['rsi'].shift(1)) &  # RSI turning down
                (df['rsi'].shift(1) > df['rsi'].shift(2))  # Confirmed turn
        )

        # Centerline crossover with trend
        centerline_buy = (
                df['rsi_cross_up_50'] &
                (df['close'] > df['close'].rolling(window=20).mean())
        )

        centerline_sell = (
                df['rsi_cross_down_50'] &
                (df['close'] < df['close'].rolling(window=20).mean())
        )

        # Divergence signals
        divergence_buy = (
                df['bullish_divergence'] &
                (df['rsi'] < 50)  # Stronger signal when below centerline
        )

        divergence_sell = (
                df['bearish_divergence'] &
                (df['rsi'] > 50)  # Stronger signal when above centerline
        )

        # Combine signals
        signals = np.where(divergence_buy | oversold_buy | centerline_buy, 1,
                           np.where(divergence_sell | overbought_sell | centerline_sell, -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class StochasticStrategy(TradingStrategy):
    """Stochastic Oscillator strategy"""

    def __init__(self, k_period: int = 14, d_period: int = 3, slowing: int = 3):
        super().__init__("Stochastic")
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate Stochastic oscillator
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'].values, df['low'].values, df['close'].values,
            fastk_period=self.k_period,
            slowk_period=self.slowing,
            slowk_matype=0,
            slowd_period=self.d_period,
            slowd_matype=0
        )

        # Stochastic conditions
        df['stoch_oversold'] = df['stoch_k'] < 20
        df['stoch_overbought'] = df['stoch_k'] > 80

        # Stochastic crossovers
        df['stoch_cross_up'] = (
                (df['stoch_k'] > df['stoch_d']) &
                (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        )

        df['stoch_cross_down'] = (
                (df['stoch_k'] < df['stoch_d']) &
                (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        )

        # Bull/Bear setups
        df['bull_setup'] = (
                df['stoch_oversold'] &
                (df['stoch_k'] > df['stoch_k'].shift(1)) &
                (df['stoch_d'] > df['stoch_d'].shift(1))
        )

        df['bear_setup'] = (
                df['stoch_overbought'] &
                (df['stoch_k'] < df['stoch_k'].shift(1)) &
                (df['stoch_d'] < df['stoch_d'].shift(1))
        )

        # Hidden divergence
        df['price_uptrend'] = df['close'] > df['close'].rolling(window=20).mean()
        df['price_downtrend'] = df['close'] < df['close'].rolling(window=20).mean()

        df['hidden_bull_div'] = (
                (df['close'] > df['close'].shift(3)) &
                (df['stoch_k'] < df['stoch_k'].shift(3)) &
                df['price_uptrend']
        )

        df['hidden_bear_div'] = (
                (df['close'] < df['close'].shift(3)) &
                (df['stoch_k'] > df['stoch_k'].shift(3)) &
                df['price_downtrend']
        )

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Oversold/Overbought with crossover
        oversold_buy = (
                df['stoch_oversold'] &
                df['stoch_cross_up']
        )

        overbought_sell = (
                df['stoch_overbought'] &
                df['stoch_cross_down']
        )

        # Bull/Bear setups
        bull_setup_buy = (
                df['bull_setup'] &
                (df['stoch_k'] > df['stoch_k'].shift(1))
        )

        bear_setup_sell = (
                df['bear_setup'] &
                (df['stoch_k'] < df['stoch_k'].shift(1))
        )

        # Hidden divergence signals
        hidden_div_buy = df['hidden_bull_div']
        hidden_div_sell = df['hidden_bear_div']

        # Combine signals
        signals = np.where(oversold_buy | bull_setup_buy | hidden_div_buy, 1,
                           np.where(overbought_sell | bear_setup_sell | hidden_div_sell, -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class IchimokuStrategy(TradingStrategy):
    """Ichimoku Cloud strategy"""

    def __init__(
            self,
            tenkan_period: int = 9,
            kijun_period: int = 26,
            senkou_span_b_period: int = 52,
            displacement: int = 26
    ):
        super().__init__("Ichimoku")
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.displacement = displacement

    def midpoint(self, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """Calculate midpoint for a given period"""
        return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate Tenkan-sen (Conversion Line)
        df['tenkan_sen'] = self.midpoint(df['high'], df['low'], self.tenkan_period)

        # Calculate Kijun-sen (Base Line)
        df['kijun_sen'] = self.midpoint(df['high'], df['low'], self.kijun_period)

        # Calculate Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.displacement)

        # Calculate Senkou Span B (Leading Span B)
        df['senkou_span_b'] = self.midpoint(df['high'], df['low'], self.senkou_span_b_period).shift(self.displacement)

        # Calculate Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-self.displacement)

        # Calculate Cloud direction
        df['cloud_direction'] = np.where(df['senkou_span_a'] > df['senkou_span_b'], 1,
                                         np.where(df['senkou_span_a'] < df['senkou_span_b'], -1, 0))

        # Calculate Cloud thickness
        df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])

        # TK Cross
        df['tk_cross_up'] = (
                (df['tenkan_sen'] > df['kijun_sen']) &
                (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))
        )

        df['tk_cross_down'] = (
                (df['tenkan_sen'] < df['kijun_sen']) &
                (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1))
        )

        # Price relative to Cloud
        df['price_above_cloud'] = (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])
        df['price_below_cloud'] = (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])
        df['price_in_cloud'] = ~(df['price_above_cloud'] | df['price_below_cloud'])

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Strong buy signals
        strong_buy = (
                df['tk_cross_up'] &
                df['price_above_cloud'] &
                (df['cloud_direction'] == 1) &
                (df['close'] > df['kijun_sen'])
        )

        # Strong sell signals
        strong_sell = (
                df['tk_cross_down'] &
                df['price_below_cloud'] &
                (df['cloud_direction'] == -1) &
                (df['close'] < df['kijun_sen'])
        )

        # Weaker buy signals
        weak_buy = (
                df['tk_cross_up'] &
                df['price_in_cloud'] &
                (df['cloud_direction'] == 1)
        )

        # Weaker sell signals
        weak_sell = (
                df['tk_cross_down'] &
                df['price_in_cloud'] &
                (df['cloud_direction'] == -1)
        )

        # Combine signals with priority to strong signals
        signals = np.where(strong_buy, 1,
                           np.where(strong_sell, -1,
                                    np.where(weak_buy, 1,
                                             np.where(weak_sell, -1, 0))))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class OrderFlowStrategy(TradingStrategy):
    """Order Flow analysis strategy (simulated with available data)"""

    def __init__(self):
        super().__init__("Order Flow")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Note: True order flow analysis requires tick data with bid/ask volumes
        # Here we'll simulate some order flow metrics using available OHLCV data

        # Calculate delta (close - open) as proxy for buying/selling pressure
        df['delta'] = df['close'] - df['open']
        df['delta_ratio'] = df['delta'] / (df['high'] - df['low']).replace(0, 0.001)

        # Calculate relative volume
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['avg_volume']

        # Calculate aggressive buying/selling volume (approximation)
        df['buying_volume'] = np.where(df['close'] > df['open'], df['volume'],
                                       df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low']))
        df['selling_volume'] = np.where(df['close'] < df['open'], df['volume'],
                                        df['volume'] * (df['high'] - df['close']) / (df['high'] - df['low']))

        # Volume-weighted price movement
        df['volume_price_trend'] = df['delta'] * df['relative_volume']

        # Cumulative delta
        df['cum_delta'] = df['delta'].cumsum()
        df['cum_delta_change'] = df['cum_delta'].diff(3)

        # Volume profile zones (approximation)
        df['session_high_volume_zone'] = np.where(
            df['volume'] > df['volume'].rolling(window=10).quantile(0.8),
            (df['high'] + df['low']) / 2,
            np.nan
        ).ffill()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Buy signals - strong buying pressure
        buy_pressure = (
                (df['delta'] > 0) &
                (df['delta_ratio'] > 0.6) &
                (df['relative_volume'] > 1.2) &
                (df['buying_volume'] > df['selling_volume'] * 1.5) &
                (df['cum_delta_change'] > 0)
        )

        # Sell signals - strong selling pressure
        sell_pressure = (
                (df['delta'] < 0) &
                (df['delta_ratio'] < -0.6) &
                (df['relative_volume'] > 1.2) &
                (df['selling_volume'] > df['buying_volume'] * 1.5) &
                (df['cum_delta_change'] < 0)
        )

        # Generate signals
        signals = np.where(buy_pressure, 1, np.where(sell_pressure, -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class MarketProfileStrategy(TradingStrategy):
    """Market Profile / TPO based strategy"""

    def __init__(self, num_bins: int = 30):
        super().__init__("Market Profile")
        self.num_bins = num_bins

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Add date column for grouping
        df['date'] = df.index.date

        # Calculate Value Area for each day (approximation)
        for date, group in df.groupby('date'):
            if len(group) < 5:
                continue

            # Define price range for the day
            day_high = group['high'].max()
            day_low = group['low'].min()

            # Create price bins
            if day_high == day_low:
                continue

            price_bins = np.linspace(day_low, day_high, self.num_bins + 1)
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2

            # Calculate TPO count for each bin (approximation using OHLC data)
            tpo_counts = np.zeros(self.num_bins)

            for idx, row in group.iterrows():
                # Which bins does this candle span?
                candle_bins = np.where(
                    (bin_centers >= row['low']) & (bin_centers <= row['high'])
                )[0]

                # Add TPO counts weighted by volume
                tpo_counts[candle_bins] += row['volume'] / len(candle_bins) if len(candle_bins) > 0 else 0

            # Find Point of Control (POC) - price with highest TPO count
            if len(tpo_counts) > 0:
                poc_idx = np.argmax(tpo_counts)
                poc_price = bin_centers[poc_idx]

                # Calculate Value Area (70% of total TPO count)
                total_tpo = np.sum(tpo_counts)
                target_tpo = 0.7 * total_tpo

                # Sort by TPO count
                sorted_idx = np.argsort(tpo_counts)[::-1]

                # Take bins until we reach target
                cum_tpo = 0
                value_area_bins = []

                for idx in sorted_idx:
                    value_area_bins.append(idx)
                    cum_tpo += tpo_counts[idx]
                    if cum_tpo >= target_tpo:
                        break

                # Define Value Area High and Low
                if value_area_bins:
                    va_high = bin_centers[max(value_area_bins)]
                    va_low = bin_centers[min(value_area_bins)]

                    # Assign to all rows in the day
                    df.loc[group.index, 'poc'] = poc_price
                    df.loc[group.index, 'va_high'] = va_high
                    df.loc[group.index, 'va_low'] = va_low

        # Calculate distance from POC and Value Area
        df['dist_from_poc'] = (df['close'] - df['poc']) / df['close'] * 100
        df['dist_from_va_high'] = (df['va_high'] - df['close']) / df['close'] * 100
        df['dist_from_va_low'] = (df['close'] - df['va_low']) / df['close'] * 100

        # Drop the date column
        df = df.drop('date', axis=1)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # Buy signals - rejection from Value Area Low
        va_low_rejection = (
                (df['low'] < df['va_low']) &
                (df['close'] > df['va_low']) &
                (df['dist_from_va_low'] < 0.2) &  # Close to VA Low
                (df['dist_from_va_low'].shift(1) < 0)  # Previously below VA Low
        )

        # Sell signals - rejection from Value Area High
        va_high_rejection = (
                (df['high'] > df['va_high']) &
                (df['close'] < df['va_high']) &
                (df['dist_from_va_high'] < 0.2) &  # Close to VA High
                (df['dist_from_va_high'].shift(1) < 0)  # Previously above VA High
        )

        # Breakout signals
        breakout_buy = (
                (df['close'] > df['va_high']) &
                (df['close'].shift(1) <= df['va_high'].shift(1)) &
                (df['volume'] > df['volume'].rolling(window=10).mean())  # Volume confirmation
        )

        breakout_sell = (
                (df['close'] < df['va_low']) &
                (df['close'].shift(1) >= df['va_low'].shift(1)) &
                (df['volume'] > df['volume'].rolling(window=10).mean())  # Volume confirmation
        )

        # Generate signals with priority to breakouts
        signals = np.where(breakout_buy, 1,
                           np.where(breakout_sell, -1,
                                    np.where(va_low_rejection, 1,
                                             np.where(va_high_rejection, -1, 0))))

        # Convert to pandas Series
        signals = pd.Series(signals, index=df.index)

        return signals


class DeepLearningStrategy(TradingStrategy):
    """Strategy using deep learning predictions"""

    def __init__(self, lookback_period: int = 20):
        super().__init__("Deep Learning")
        self.lookback_period = lookback_period
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.features = None

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for deep learning model"""
        df = data.copy()

        # Simple feature engineering for deep learning
        features = []

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        for i in [3, 5, 10, 20]:
            df[f'return_{i}d'] = df['returns'].rolling(window=i).sum()
            df[f'std_{i}d'] = df['returns'].rolling(window=i).std()
            df[f'max_high_{i}d'] = df['high'].rolling(window=i).max() / df['close'] - 1
            df[f'min_low_{i}d'] = df['close'] / df['low'].rolling(window=i).min() - 1

        # Technical indicators
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        df['rsi_5'] = talib.RSI(df['close'].values, timeperiod=5)

        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
            df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )

        df['atr'] = talib.ATR(
            df['high'].values, df['low'].values, df['close'].values, timeperiod=14
        )

        # Volume-based features
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

        # Select features
        feature_columns = [
            'returns', 'log_returns', 'return_3d', 'return_5d', 'return_10d',
            'std_5d', 'std_20d', 'max_high_5d', 'min_low_5d', 'rsi', 'rsi_5',
            'macd', 'macdsignal', 'macdhist', 'atr', 'volume_ratio', 'volatility'
        ]

        # Store feature names
        self.features = feature_columns

        # Drop NaN values
        df = df.dropna()

        # Extract features
        if len(df) > 0:
            features = df[feature_columns].values

        return features

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Prepare features
        features = self.prepare_features(df)

        # Add prediction column
        df['dl_prediction'] = np.nan

        # If model is trained, make predictions
        if self.is_trained and len(features) > 0:
            try:
                # Scale features
                scaled_features = self.scaler.transform(features)

                # Reshape for LSTM if needed
                if hasattr(self.model, 'layers') and any('lstm' in str(layer).lower() for layer in self.model.layers):
                    # Reshape to [samples, time steps, features]
                    reshaped_features = []
                    for i in range(self.lookback_period, len(scaled_features)):
                        reshaped_features.append(scaled_features[i - self.lookback_period:i])

                    if reshaped_features:
                        reshaped_features = np.array(reshaped_features)
                        predictions = self.model.predict(reshaped_features)

                        # Map predictions to original index
                        valid_indices = df.index[self.lookback_period:len(reshaped_features) + self.lookback_period]
                        for i, idx in enumerate(valid_indices):
                            df.at[idx, 'dl_prediction'] = predictions[i]
                else:
                    # For non-sequence models
                    predictions = self.model.predict(scaled_features)
                    valid_indices = df.dropna().index[-len(predictions):]
                    for i, idx in enumerate(valid_indices):
                        df.at[idx, 'dl_prediction'] = predictions[i]

            except Exception as e:
                logger.error(f"Error making predictions: {e}")

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate_features(data)

        signals = pd.Series(0, index=df.index)

        # If model is trained, use predictions
        if self.is_trained:
            # Generate signals based on model predictions
            # Assuming prediction is probability of price increase
            signals = np.where(df['dl_prediction'] > 0.7, 1,
                               np.where(df['dl_prediction'] < 0.3, -1, 0))

            # Filter out NaN values
            signals = pd.Series(signals, index=df.index).fillna(0)

        return signals


class EnsembleStrategy(TradingStrategy):
    """Ensemble strategy combining multiple individual strategies"""

    def __init__(self, strategies: Dict[TradingStrategy, float] = None):
        super().__init__("Ensemble")
        self.strategies = strategies or {}

    def add_strategy(self, strategy: TradingStrategy, weight: float = 1.0):
        """Add a strategy to the ensemble"""
        self.strategies[strategy] = weight
        return self

    def normalize_weights(self):
        """Normalize strategy weights to sum to 1"""
        total_weight = sum(self.strategies.values())
        if total_weight > 0:
            for strategy in self.strategies:
                self.strategies[strategy] /= total_weight

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features using all strategies in the ensemble"""
        df = data.copy()

        # Calculate features for each strategy
        for strategy in self.strategies:
            try:
                strategy_features = strategy.calculate_features(data)

                # Add strategy name as prefix to avoid column name conflicts
                prefix = f"{strategy.name.lower().replace(' ', '_')}_"

                for col in strategy_features.columns:
                    if col not in data.columns:  # Skip original columns
                        df[f"{prefix}{col}"] = strategy_features[col]
            except Exception as e:
                logger.error(f"Error calculating features for {strategy.name}: {e}")

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate weighted signals from all strategies"""
        signals = pd.Series(0, index=data.index)

        # Normalize weights
        self.normalize_weights()

        # Generate signals for each strategy
        strategy_signals = {}
        for strategy, weight in self.strategies.items():
            try:
                strategy_signals[strategy.name] = strategy.generate_signals(data) * weight
            except Exception as e:
                logger.error(f"Error generating signals for {strategy.name}: {e}")
                strategy_signals[strategy.name] = pd.Series(0, index=data.index)

        # Combine signals
        for name, signal in strategy_signals.items():
            signals = signals.add(signal, fill_value=0)

        # Apply threshold for final signals
        signals = np.where(signals > 0.2, 1, np.where(signals < -0.2, -1, 0))

        # Convert to pandas Series
        signals = pd.Series(signals, index=data.index)

        return signals


# Factory method to create all strategies
def create_strategies() -> Dict[str, TradingStrategy]:
    """Create all trading strategies with default parameters"""
    strategies = {
        "trend_following": TrendFollowingStrategy(),
        "mean_reversion": MeanReversionStrategy(),
        "breakout": BreakoutStrategy(),
        "vwap": VWAPStrategy(),
        "support_resistance": SupportResistanceStrategy(),
        "momentum": MomentumStrategy(),
        "volume_profile": VolumeProfileStrategy(),
        "bollinger_bands": BollingerBandStrategy(),
        "macd": MACDStrategy(),
        "rsi": RSIStrategy(),
        "stochastic": StochasticStrategy(),
        "ichimoku": IchimokuStrategy(),
        "order_flow": OrderFlowStrategy(),
        "market_profile": MarketProfileStrategy(),
        "deep_learning": DeepLearningStrategy()
    }

    # Create ensemble strategy with all other strategies
    ensemble = EnsembleStrategy()
    for name, strategy in strategies.items():
        weight = config.STRATEGY_WEIGHTS.get(name, 1.0 / len(strategies))
        ensemble.add_strategy(strategy, weight)

    strategies["ensemble"] = ensemble

    return strategies


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100)
    }, index=dates)

    # Ensure high >= open, close, low
    data['high'] = data[['high', 'open', 'close']].max(axis=1) + 0.1

    # Ensure low <= open, close, high
    data['low'] = data[['low', 'open', 'close']].min(axis=1) - 0.1

    # Create strategies
    all_strategies = create_strategies()

    # Generate signals for each strategy
    for name, strategy in all_strategies.items():
        signals = strategy.generate_signals(data)
        print(f"{name} Strategy Signals: {signals.value_counts().to_dict()}")