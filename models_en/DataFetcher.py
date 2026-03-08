"""
Data fetching module for option/equity data.
Fetches tick data via RiceQuant (Miqiang) API.
"""

import os
import rqdatac as rq
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import datetime
import warnings
warnings.filterwarnings('ignore')

# RiceQuant API license is read from env var RQDATAC_LICENSE to avoid hardcoding secrets.
# Before use: export RQDATAC_LICENSE='your_license_key'


class DataFetcher():
    
    def __init__(self, license: Optional[str] = None):
        """
        Initialize the data fetcher.
        
        Args:
            license: RiceQuant API license key. If not provided, must be passed later via init_connection().
        """
        self.license = license
        self.is_connected = False

    def init_connection(self, license: Optional[str] = None):
        """
        Initialize RiceQuant data connection.
        
        Args:
            license: RiceQuant API license key. If not provided, reads from env var RQDATAC_LICENSE.
        """
        if license:
            self.license = license
        elif not self.license:
            self.license = os.environ.get('RQDATAC_LICENSE')
            
        if not self.license:
            raise ValueError(
                "RiceQuant API license key required. Pass via init_connection(license='...'), "
                "or set env var: export RQDATAC_LICENSE='your_license_key'"
            )
            
        try:
            rq.init('license', self.license)
            self.is_connected = True
            print("RiceQuant connection established.")
        except Exception as e:
            print(f"RiceQuant connection failed: {e}")
            raise
    
    def get_trading_days(self, start_date: str, end_date: str, market: str = 'cn') -> List[str]:
        """
        Get trading days within the specified date range.
        
        Args:
            start_date: Start date, format 'YYYYMMDD'
            end_date: End date, format 'YYYYMMDD'
            market: Market code, default 'cn' (China)
        
        Returns:
            List of trading day strings
        """
        if not self.is_connected:
            raise RuntimeError("Call init_connection() first")
        
        try:
            trading_days = rq.get_trading_dates(start_date, end_date, market=market)
            trading_days_str = [day.strftime('%Y%m%d') for day in trading_days]
            return trading_days_str
        except Exception as e:
            print(f"Failed to fetch trading days: {e}")
            raise
    
    def get_tick_data(
        self, 
        symbol: str, 
        date: str,
        start_time: datetime.time = datetime.time(hour=10, minute=0),
        end_time: datetime.time = datetime.time(hour=11, minute=0)
    ) -> pd.DataFrame:
        """
        Get tick data for a symbol on a specific date and time window.
        
        Args:
            symbol: Symbol code, e.g. '000300.XSHG'
            date: Trading date, format 'YYYYMMDD'
            start_time: Start of time window
            end_time: End of time window
        
        Returns:
            DataFrame with tick data
        """
        if not self.is_connected:
            raise RuntimeError("Call init_connection() first")
        
        try:
            df = rq.get_price(
                symbol,
                start_date=date,
                end_date=date,
                frequency='tick',
                time_slice=(start_time, end_time)
            )
            return df
        except Exception as e:
            print(f"Failed to fetch tick data for {date}: {e}")
            return pd.DataFrame()
    
    def get_tick_data_batch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        start_time: datetime.time = datetime.time(hour=10, minute=0),
        end_time: datetime.time = datetime.time(hour=11, minute=0),
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Batch fetch tick data over a date range.
        
        Args:
            symbol: Symbol code
            start_date: Start date, format 'YYYYMMDD'
            end_date: End date, format 'YYYYMMDD'
            start_time: Daily window start
            end_time: Daily window end
            verbose: Whether to print progress
        
        Returns:
            Concatenated DataFrame
        """
        trading_days = self.get_trading_days(start_date, end_date)
        
        all_data = []
        failed_days = []
        
        for i, day in enumerate(trading_days):
            if verbose and (i % 10 == 0 or i == len(trading_days) - 1):
                print(f"Fetching: {i+1}/{len(trading_days)} - {day}")
            
            df = self.get_tick_data(symbol, day, start_time, end_time)
            
            if not df.empty:
                all_data.append(df)
            else:
                failed_days.append(day)
        
        if verbose:
            print(f"Successfully fetched {len(all_data)}/{len(trading_days)} trading days")
            if failed_days:
                print(f"Failed dates: {failed_days}")
        
        if all_data:
            return pd.concat(all_data, axis=0)
        else:
            return pd.DataFrame()
