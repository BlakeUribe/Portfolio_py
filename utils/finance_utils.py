import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from numpy.typing import NDArray

# If Updated remember to adjust date at bottom


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

def get_sector(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('sector', None)  # Get the sector, return None if not available
    except Exception as e:
        print(f"Error retrieving sector for {ticker}: {e}")
        return None



print('\n---------------------------------')
print('finance_utils.py successfully loaded, updated last Jan. 15 2025')
print('---------------------------------')
print('\n')