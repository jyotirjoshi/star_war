import pandas as pd
import numpy as np
import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional
import logging
from bs4 import BeautifulSoup
import websocket
import json
import threading
import time
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.RESULTS_DIR, "data_handler.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataHandler")


class DataHandler:
    """
    Manages data acquisition, processing, and storage for the NQ E-mini trading bot
    """

    def __init__(self):
        self.historical_data = None
        self.live_data_buffer = pd.DataFrame()
        self.live_data_callback = None
        self.live_data_stream = None
        self.is_streaming = False
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY",
                                           config.ALPHA_VANTAGE_KEY if hasattr(config, "ALPHA_VANTAGE_KEY") else "")

    def load_historical_data(
            self,
            source: str = "local_csv",
            path: str = None,
            start_date: str = None,
            end_date: str = None,
            timeframe: str = "5min"
    ) -> pd.DataFrame:
        """
        Load historical price data from various sources
        """
        if source == "local_csv":
            path = path or config.HISTORICAL_DATA_PATH
            logger.info(f"Loading historical data from CSV: {path}")
            try:
                df = pd.read_csv(path)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                logger.info(f"Loaded {len(df)} rows of historical data")
                self.historical_data = df
                return df
            except Exception as e:
                logger.error(f"Error loading CSV data: {e}")
                return pd.DataFrame()

        elif source == "alpha_vantage":
            if not self.alpha_vantage_key:
                logger.error("Alpha Vantage API key not provided")
                return pd.DataFrame()

            logger.info(f"Fetching historical data from Alpha Vantage API")
            try:
                # Map timeframe to Alpha Vantage interval
                interval_map = {
                    "1min": "1min",
                    "5min": "5min",
                    "15min": "15min",
                    "30min": "30min",
                    "60min": "60min",
                    "1h": "60min",
                    "1d": "daily"
                }

                av_interval = interval_map.get(timeframe, "5min")

                # For daily data
                if av_interval == "daily":
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NQ&outputsize=full&apikey={self.alpha_vantage_key}"
                    data_key = "Time Series (Daily)"
                else:
                    # For intraday data
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NQ&interval={av_interval}&outputsize=full&apikey={self.alpha_vantage_key}"
                    data_key = f"Time Series ({av_interval})"

                response = requests.get(url)
                data = response.json()

                if data_key not in data:
                    logger.error(f"Error in Alpha Vantage response: {data}")
                    return pd.DataFrame()

                # Parse the JSON response
                time_series = data[data_key]
                records = []

                for timestamp, values in time_series.items():
                    records.append({
                        'time': timestamp,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': float(values['5. volume'])
                    })

                df = pd.DataFrame(records)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df = df.sort_index()

                # Filter by date if specified
                if start_date:
                    start_date = pd.to_datetime(start_date)
                    df = df[df.index >= start_date]

                if end_date:
                    end_date = pd.to_datetime(end_date)
                    df = df[df.index <= end_date]

                logger.info(f"Loaded {len(df)} rows from Alpha Vantage API")
                self.historical_data = df
                return df

            except Exception as e:
                logger.error(f"Error fetching data from Alpha Vantage: {e}")
                return pd.DataFrame()

        elif source == "yfinance":
            logger.info(f"Fetching historical data from Yahoo Finance")
            try:
                # Map timeframe to yfinance interval
                tf_map = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
                          "1h": "1h", "1d": "1d"}
                yf_tf = tf_map.get(timeframe, "5m")

                # Use NQ=F for E-mini Nasdaq futures
                symbol = "NQ=F"

                # Set default dates if not provided
                if not end_date:
                    end_date = datetime.now()
                else:
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")

                if not start_date:
                    start_date = end_date - timedelta(days=365)
                else:
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")

                df = yf.download(symbol, start=start_date, end=end_date, interval=yf_tf)

                if len(df) > 0:
                    # Rename columns to match our format
                    df = df.rename(columns={
                        'Open': 'open', 'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Volume': 'volume'
                    })
                    logger.info(f"Loaded {len(df)} rows from Yahoo Finance")
                    self.historical_data = df
                    return df
                else:
                    logger.warning("No data returned from Yahoo Finance")
                    return pd.DataFrame()

            except Exception as e:
                logger.error(f"Error fetching data from Yahoo Finance: {e}")
                return pd.DataFrame()

        else:
            logger.error(f"Unsupported data source: {source}")
            return pd.DataFrame()

    def preprocess_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean and prepare data for analysis
        """
        df = data if data is not None else self.historical_data

        if df is None or len(df) == 0:
            logger.error("No data to preprocess")
            return pd.DataFrame()

        logger.info("Preprocessing data...")

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            else:
                logger.warning("No time column found, using numeric index")

        # Handle missing values
        if df.isna().any().any():
            logger.info(f"Filling {df.isna().sum().sum()} missing values")
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Check for duplicated timestamps
        if df.index.duplicated().any():
            logger.warning(f"Found {df.index.duplicated().sum()} duplicate timestamps, keeping last value")
            df = df[~df.index.duplicated(keep='last')]

        # Sort by timestamp
        df = df.sort_index()

        # Add basic returns
        df['returns'] = df['close'].pct_change()

        logger.info(f"Preprocessing complete, data shape: {df.shape}")
        return df

    def start_live_data_stream(self, callback=None):
        """
        Start streaming live market data
        Note: Alpha Vantage does not provide true streaming API but we can simulate with polling
        """
        if self.is_streaming:
            logger.warning("Live data stream is already running")
            return

        if not self.alpha_vantage_key:
            logger.error("Alpha Vantage API key not provided")
            return

        self.live_data_callback = callback
        self.is_streaming = True

        # Start polling in a separate thread
        def polling_thread():
            logger.info("Starting Alpha Vantage polling for live data")
            last_timestamp = None

            while self.is_streaming:
                try:
                    # Poll Alpha Vantage for the latest data (1-min interval)
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NQ&interval=1min&outputsize=compact&apikey={self.alpha_vantage_key}"
                    response = requests.get(url)
                    data = response.json()

                    if "Time Series (1min)" in data:
                        time_series = data["Time Series (1min)"]
                        # Get the most recent data point
                        latest_timestamp = list(time_series.keys())[0]
                        latest_timestamp_dt = pd.to_datetime(latest_timestamp)

                        # Only process if this is a new timestamp
                        if last_timestamp is None or latest_timestamp_dt > last_timestamp:
                            values = time_series[latest_timestamp]
                            tick = {
                                'time': latest_timestamp_dt,
                                'open': float(values['1. open']),
                                'high': float(values['2. high']),
                                'low': float(values['3. low']),
                                'close': float(values['4. close']),
                                'volume': float(values['5. volume'])
                            }

                            # Add to buffer
                            self.live_data_buffer = self.live_data_buffer.append(
                                pd.Series(tick, name=tick['time']))

                            # Call the callback if provided
                            if self.live_data_callback:
                                self.live_data_callback(tick)

                            last_timestamp = latest_timestamp_dt
                            logger.debug(f"Received new tick: {latest_timestamp}")

                    # Sleep to respect API rate limits (Alpha Vantage allows 5 requests per minute for free tier)
                    time.sleep(15)  # Sleep for 15 seconds between requests

                except Exception as e:
                    logger.error(f"Error polling Alpha Vantage: {e}")
                    time.sleep(30)  # Wait longer on error

        # Start polling thread
        self.live_data_stream = threading.Thread(target=polling_thread, daemon=True)
        self.live_data_stream.start()
        logger.info("Started Alpha Vantage polling thread")

    def stop_live_data_stream(self):
        """
        Stop the live data stream
        """
        if not self.is_streaming:
            logger.warning("No live data stream is running")
            return

        self.is_streaming = False
        logger.info("Stopping Alpha Vantage polling")

        # Wait for thread to terminate
        if self.live_data_stream and self.live_data_stream.is_alive():
            self.live_data_stream.join(timeout=2.0)

    def get_free_data_sources(self) -> Dict[str, str]:
        """
        Return a list of available free data sources for NQ E-mini futures
        """
        return {
            "alpha_vantage": "Alpha Vantage (free tier with API key)",
            "yfinance": "Yahoo Finance (limited history and resolution)",
            "twelvedata": "Twelve Data (limited free tier)",
            "iex": "IEX Cloud (limited free tier)",
            "alternative_data": "Web scraping strategies (educational purpose only)"
        }

    def save_data(self, data: pd.DataFrame, filename: str):
        """
        Save data to disk
        """
        try:
            filepath = os.path.join(config.DATA_DIR, filename)
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from disk
        """
        try:
            filepath = os.path.join(config.DATA_DIR, filename)
            data = pd.read_csv(filepath, index_col=0)
            data.index = pd.to_datetime(data.index)
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def get_alternative_data(self) -> pd.DataFrame:
        """
        Get alternative data through web scraping (for educational purposes)
        """
        logger.warning("Alternative data collection is for educational purposes only")
        logger.warning("Always comply with websites' terms of service and legal requirements")

        # This is a placeholder educational example
        try:
            # Example: Get economic calendar data (this would need to comply with the source's terms)
            url = "https://www.investing.com/economic-calendar/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # This is just a simple example and would need to be customized
                events = []

                for event in soup.select('#economicCalendarData .js-event-item'):
                    try:
                        time_element = event.select_one('.time')
                        event_element = event.select_one('.event')
                        impact_element = event.select_one('.sentiment')

                        if time_element and event_element:
                            events.append({
                                'time': time_element.text.strip(),
                                'event': event_element.text.strip(),
                                'impact': impact_element['title'] if impact_element else 'Unknown'
                            })
                    except Exception as e:
                        continue

                return pd.DataFrame(events)
            else:
                logger.error(f"Failed to get alternative data, status code: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting alternative data: {e}")
            return pd.DataFrame()

    def get_data_stats(self, data: pd.DataFrame = None) -> Dict:
        """
        Return basic statistics about the data
        """
        df = data if data is not None else self.historical_data

        if df is None or len(df) == 0:
            logger.error("No data to analyze")
            return {}

        stats = {
            "start_date": df.index.min().strftime("%Y-%m-%d"),
            "end_date": df.index.max().strftime("%Y-%m-%d"),
            "trading_days": len(np.unique(df.index.date)),
            "total_rows": len(df),
            "avg_daily_range": (df['high'] - df['low']).mean(),
            "avg_daily_volume": df.groupby(df.index.date)['volume'].sum().mean(),
            "price_std": df['close'].std(),
            "avg_returns": df['close'].pct_change().mean() * 100,
            "max_return": df['close'].pct_change().max() * 100,
            "min_return": df['close'].pct_change().min() * 100,
        }

        return stats


# Example usage
if __name__ == "__main__":
    data_handler = DataHandler()

    # Load historical data
    data = data_handler.load_historical_data()

    if len(data) > 0:
        # Preprocess data
        processed_data = data_handler.preprocess_data(data)

        # Print some statistics
        stats = data_handler.get_data_stats(processed_data)
        for key, value in stats.items():
            print(f"{key}: {value}")