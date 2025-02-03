import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from numpy.typing import NDArray

# If Updated remember to adjust date at bottom

def get_corr_pairs_of_stocks(tickers: list) -> pd.DataFrame:
    """
    Computes and returns the correlation pairs of given stock tickers.

    Args:
        tickers (list): A list of stock tickers.

    Returns:
        pd.DataFrame: A DataFrame containing correlation pairs sorted in ascending order.
    """
    
    # Download adjusted closing prices for the past year
    grouped_sector_data = yf.download(tickers, period='1y', auto_adjust=True)['Close']
    correlation_matrix = grouped_sector_data.corr()
    corr_pairs = correlation_matrix.stack()

    # Remove self-correlations (where ticker_1 == ticker_2)
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]

    # Format index to be "Ticker1_Ticker2"
    corr_pairs.index = corr_pairs.index.map('_'.join)
    
    # Convert to DataFrame and rename columns
    corr_df = corr_pairs.to_frame(name='Correlation')

    # Drop duplicate correlation values (since correlation is symmetric)
    corr_df = corr_df.drop_duplicates(subset=['Correlation'])

    # Sort by correlation value in ascending order
    return corr_df.sort_values(by='Correlation', ascending=True)


def get_top_n_by_sector(df: pd.DataFrame, filter_var: str, top_n: int = 3) -> pd.DataFrame:
    """
    Returns the top N companies per sector based on a specified financial metric.

    Args:
        df (pd.DataFrame): DataFrame containing stock valuation data, including 'sector'.
        filter_var (str): The column name to filter by (e.g., 'profitMargins').
        top_n (int, optional): Number of top companies to return per sector. Defaults to 3.

    Returns:
        pd.DataFrame: DataFrame with the top N companies per sector sorted by the given metric.
    """
    return df.groupby('sector').apply(lambda x: x.nlargest(top_n, filter_var)).reset_index(drop=True)


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
print('finance_utils.py successfully loaded, updated last Jan. 27 2025')
print('---------------------------------')
print('\n')