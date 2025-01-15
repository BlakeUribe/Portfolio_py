import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from numpy.typing import NDArray




# Vectorized function to calculate Sharpe ratio for multiple tickers
def calculate_sharpe_ratio(tickers: NDArray, tbill: pd.DataFrame, start_date: datetime, end_date: datetime) -> float:
    # Download stock data for all tickers at once to minimize repeated API calls
    stock_data = yf.download(tickers.tolist(), start=start_date, end=end_date, auto_adjust=True)['Close']

    # Calculate daily returns for all tickers
    daily_returns = stock_data.pct_change()

    # Drop NaNs and align T-Bill data
    stock_data_clean = daily_returns.dropna()
    tbill_aligned = tbill.reindex(stock_data_clean.index, method='ffill')

    # Calculate excess returns by subtracting the T-Bill rate
    excess_returns = stock_data_clean.sub(tbill_aligned['^IRX'], axis=0)

    # Calculate Sharpe ratio for each ticker using vectorized operations
    excess_returns_std = excess_returns.std()
    average_excess_daily_ret = excess_returns.mean()

    daily_sharpe_ratio = average_excess_daily_ret / excess_returns_std
    annualized_sharpe = daily_sharpe_ratio * np.sqrt(360)

    return annualized_sharpe




# # this funtion may not be in use
# def filter_stocks(df: pd.DataFrame, 
#                  min_market_cap: float = 100_000_000_000,  # 100B
#                  min_profit_margin: float = 0.2,           # 20%
#                  min_sharpe: float = 0.75) -> pd.DataFrame:
    
#     # Handle missing values
#     filtered_df = df.dropna(subset=['marketCap', 'profitMargins', 'sharpe_ratio'])
    
#     # Create filter conditions
#     market_cap_filter = filtered_df['marketCap'] > min_market_cap
#     profit_margin_filter = filtered_df['profitMargins'] > min_profit_margin
#     sharpe_filter = filtered_df['sharpe_ratio'] > min_sharpe
    
#     # Apply filters
#     filtered_df = filtered_df[
#         market_cap_filter & 
#         profit_margin_filter & 
#         sharpe_filter
#     ]
    
#     # Add readable market cap column
#     filtered_df['marketCap_B'] = filtered_df['marketCap'] / 1_000_000_000
    
#     # Sort by market cap
#     return filtered_df.sort_values('marketCap', ascending=False)

# # Apply filters


print('\n---------------------------------')
print('finance_utils.py successfully loaded, updated last Jan. 11 2025')
print('---------------------------------')
print('\n')