import pandas as pd
import numpy as np
import os
import requests
import logging
from datetime import datetime, timedelta, timezone
import time
from typing import Dict, List, Tuple, Union, Optional
import alpaca_trade_api as tradeapi
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("AlpacaDataHandler")


class AlpacaDataHandler:
    """
    Handler for getting Nasdaq-100 E-mini futures data from Alpaca
    """

    def __init__(self):
        self.api = None
        self.connect_to_alpaca()
        self.live_data_buffer = pd.DataFrame()
        self.live_data_callback = None
        self.is_streaming = False
        self.stop_streaming = False

    def connect_to_alpaca(self):
        """Connect to Alpaca API"""
        try:
            api_key = os.getenv("ALPACA_API_KEY") or config.ALPACA_API_KEY
            api_secret = os.getenv("ALPACA_SECRET_KEY") or config.ALPACA_SECRET_KEY
            base_url = os.getenv("ALPACA_ENDPOINT") or config.ALPACA_ENDPOINT

            if not api_key or not api_secret:
                logger.error("Alpaca API credentials not found")
                return False

            self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            logger.info("Connected to Alpaca API")
            return True

        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            return False

    def get_historical_data(
            self,
            symbol: str = "NQ",
            timeframe: str = "5Min",
            start_date: str = None,
            end_date: str = None,
            limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical futures data from Alpaca

        Parameters:
        - symbol: The futures symbol (e.g., "NQ" for Nasdaq-100 E-mini)
        - timeframe: Time interval (e.g., "5Min", "1D")
        - start_date: Start date in YYYY-MM-DD format
        - end_date: End date in YYYY-MM-DD format
        - limit: Maximum number of bars to fetch

        Returns:
        - DataFrame with OHLCV data
        """
        if not self.api:
            logger.error("Alpaca API not connected")
            return pd.DataFrame()

        try:
            # Prepare parameters
            if not end_date:
                end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            if not start_date:
                # Default to 1 month of data
                start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                start = start_date

            # Format symbol for Alpaca
            formatted_symbol = f"MNQ" if symbol == "NQ" else symbol

            logger.info(f"Fetching {timeframe} bars for {formatted_symbol} from {start} to {end_date}")

            # Get historical data from Alpaca
            bars = self.api.get_bars(
                formatted_symbol,
                timeframe,
                start=start,
                end=end_date,
                limit=limit,
                adjustment='raw'
            ).df

            if len(bars) > 0:
                # Rename columns to match our format
                bars = bars.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })

                logger.info(f"Fetched {len(bars)} bars from Alpaca")
                return bars
            else:
                logger.warning("No data returned from Alpaca")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data from Alpaca: {e}")
            return pd.DataFrame()

    def start_live_data_stream(self, symbol: str = "NQ", callback=None):
        """
        Start streaming live market data from Alpaca

        Parameters:
        - symbol: The futures symbol (e.g., "NQ" for Nasdaq-100 E-mini)
        - callback: Function to call with each new data point
        """
        if not self.api:
            logger.error("Alpaca API not connected")
            return False

        if self.is_streaming:
            logger.warning("Live data stream is already running")
            return True

        self.live_data_callback = callback
        formatted_symbol = f"MNQ" if symbol == "NQ" else symbol

        try:
            # Create Alpaca streaming connection
            stream = tradeapi.Stream(
                key_id=os.getenv("ALPACA_API_KEY") or config.ALPACA_API_KEY,
                secret_key=os.getenv("ALPACA_SECRET_KEY") or config.ALPACA_SECRET_KEY,
                base_url=os.getenv("ALPACA_ENDPOINT") or config.ALPACA_ENDPOINT,
                data_feed='iex'  # or 'sip' for paid subscription
            )

            # Define callback for minute bars
            @stream.on_bar(formatted_symbol)
            async def on_bar(bar):
                # Process the bar data
                bar_dict = {
                    'time': pd.Timestamp(bar.timestamp),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }

                # Add to buffer
                self.live_data_buffer = self.live_data_buffer.append(
                    pd.Series(bar_dict, name=bar_dict['time'])
                )

                # Call the callback if provided
                if self.live_data_callback:
                    self.live_data_callback(bar_dict)

            # Start the stream
            logger.info(f"Starting Alpaca stream for {formatted_symbol}")
            self.is_streaming = True
            self.stop_streaming = False
            stream.run()

            return True

        except Exception as e:
            logger.error(f"Error starting Alpaca stream: {e}")
            self.is_streaming = False
            return False

    def stop_live_data_stream(self):
        """Stop streaming data"""
        if not self.is_streaming:
            logger.warning("No live data stream is running")
            return

        self.stop_streaming = True
        self.is_streaming = False
        logger.info("Stopped Alpaca data stream")

    def get_latest_quote(self, symbol: str = "NQ") -> Dict:
        """Get latest quote for a symbol"""
        if not self.api:
            logger.error("Alpaca API not connected")
            return {}

        try:
            # Format symbol for Alpaca
            formatted_symbol = f"MNQ" if symbol == "NQ" else symbol

            # Get latest quote
            quote = self.api.get_latest_quote(formatted_symbol)

            return {
                'symbol': symbol,
                'bid': quote.bp,
                'ask': quote.ap,
                'bid_size': quote.bs,
                'ask_size': quote.as_,
                'timestamp': pd.Timestamp(quote.t)
            }

        except Exception as e:
            logger.error(f"Error getting quote from Alpaca: {e}")
            return {}

    def save_data_to_csv(self, data: pd.DataFrame, filepath: str) -> bool:
        """Save data to CSV file"""
        try:
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False


# Example usage
def main():
    # Create data handler
    data_handler = AlpacaDataHandler()

    # Get historical data
    data = data_handler.get_historical_data(
        symbol="NQ",
        timeframe="5Min",
        start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d")
    )

    # Print sample data
    if len(data) > 0:
        print(f"Downloaded {len(data)} bars")
        print("\nSample data:")
        print(data.head())

        # Save to CSV
        data_handler.save_data_to_csv(data, "nq_data_alpaca.csv")

    # Demonstrate live data (press Ctrl+C to stop)
    def print_tick(tick):
        print(f"New tick: {tick['time']} - Price: {tick['close']}")

    print("\nStarting live data stream... (Press Ctrl+C to stop)")
    try:
        data_handler.start_live_data_stream("NQ", print_tick)
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        data_handler.stop_live_data_stream()


if __name__ == "__main__":
    main()