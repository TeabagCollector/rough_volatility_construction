"""
期权数据获取模块
通过米筐API获取期权完整数据和Greeks信息
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

# 米筐 API license 从环境变量 RQDATAC_LICENSE 读取，避免硬编码敏感信息
# 使用前请设置: export RQDATAC_LICENSE='your_license_key'


class DataFetcher():
    
    def __init__(self, license: Optional[str] = None):
        """
        初始化数据获取器
        
        Args:
            license: 米筐API的license key，如果不提供则需要后续调用init_connection时提供
            data_dir: 数据保存目录，默认为当前目录下的data文件夹
        """
        self.license = license
        self.is_connected = False

    def init_connection(self, license: Optional[str] = None):
        """
        初始化米筐数据连接
        
        Args:
            license: 米筐API的license key，若不提供则从环境变量 RQDATAC_LICENSE 读取
        """
        if license:
            self.license = license
        elif not self.license:
            self.license = os.environ.get('RQDATAC_LICENSE')
            
        if not self.license:
            raise ValueError(
                "请提供米筐API的license key。可通过 init_connection(license='...') 传入，"
                "或设置环境变量: export RQDATAC_LICENSE='your_license_key'"
            )
            
        try:
            rq.init('license', self.license)
            self.is_connected = True
            print("✓ 米筐数据连接成功")
        except Exception as e:
            print(f"✗ 米筐数据连接失败: {e}")
            raise
    
    def get_trading_days(self, start_date: str, end_date: str, market: str = 'cn') -> List[str]:
        """
        获取指定时间段内的交易日
        
        Args:
            start_date: 开始日期，格式'YYYYMMDD'
            end_date: 结束日期，格式'YYYYMMDD'
            market: 市场代码，默认为'cn'（中国市场）
        
        Returns:
            交易日列表
        """
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")
        
        try:
            trading_days = rq.get_trading_dates(start_date, end_date, market=market)
            trading_days_str = [day.strftime('%Y%m%d') for day in trading_days]
            return trading_days_str
        except Exception as e:
            print(f"✗ 获取交易日失败: {e}")
            raise
    
    def get_tick_data(
        self, 
        symbol: str, 
        date: str,
        start_time: datetime.time = datetime.time(hour=10, minute=0),
        end_time: datetime.time = datetime.time(hour=11, minute=0)
    ) -> pd.DataFrame:
        """
        获取指定标的在特定日期和时间窗口内的tick数据
        
        Args:
            symbol: 标的代码，如'000300.XSHG'
            date: 交易日期，格式'YYYYMMDD'
            start_time: 时间窗口开始时间
            end_time: 时间窗口结束时间
        
        Returns:
            包含tick数据的DataFrame
        """
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")
        
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
            print(f"✗ 获取{date}的tick数据失败: {e}")
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
        批量获取指定时间段内的tick数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期，格式'YYYYMMDD'
            end_date: 结束日期，格式'YYYYMMDD'
            start_time: 每日时间窗口开始时间
            end_time: 每日时间窗口结束时间
            verbose: 是否显示进度信息
        
        Returns:
            合并后的DataFrame
        """
        trading_days = self.get_trading_days(start_date, end_date)
        
        all_data = []
        failed_days = []
        
        for i, day in enumerate(trading_days):
            if verbose and (i % 10 == 0 or i == len(trading_days) - 1):
                print(f"正在获取数据: {i+1}/{len(trading_days)} - {day}")
            
            df = self.get_tick_data(symbol, day, start_time, end_time)
            
            if not df.empty:
                all_data.append(df)
            else:
                failed_days.append(day)
        
        if verbose:
            print(f"✓ 成功获取 {len(all_data)}/{len(trading_days)} 个交易日的数据")
            if failed_days:
                print(f"⚠ 失败的日期: {failed_days}")
        
        if all_data:
            return pd.concat(all_data, axis=0)
        else:
            return pd.DataFrame()