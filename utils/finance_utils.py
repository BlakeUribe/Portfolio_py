import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from numpy.typing import NDArray
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, before_log, after_log

# Create and configure logger
logger = logging.getLogger("backoff_logger")
logging.basicConfig(level=logging.INFO)

# If Updated remember to adjust date at bottom


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def backtest_portfolio(stocks: list, paper_val: float, weights: list, start_date: str, end_date: str) -> float:
    """
    Backtests a portfolio of stocks over a given period.

    Parameters:
        stocks (list): List of stock tickers.
        paper_val (float): Initial portfolio value.
        weights (list): Portfolio allocation weights, entered in decimal format.
        start_date (str): Start date for backtesting.
        end_date (str): End date for backtesting.

    Returns:
        float: Final portfolio value.
    """
    hpr_list = []

    for stock in stocks:
        stock_data = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)['Close'] #notice that google, or yf does not use auto-adjust
        if stock_data.empty:
            print(f"Warning: No data for {stock}")
            hpr_list.append(0)
            continue
        if stock == '^IRX':
            hpr = stock_data.mean().iloc[0]/100 # if ticker is real tbill will need to calcuate ret different
        else:
            hpr = ((stock_data.iloc[-1] - stock_data.iloc[0]) / stock_data.iloc[0]).iloc[0] #hpr is a series, so need to select the value instead
        hpr_list.append(hpr)

    # Convert to NumPy for efficient calculations
    hpr_array = np.array(hpr_list)
    weights_array = np.array(weights)
    return_on_weights = []

    # Compute final portfolio value
    for i in range(len(hpr_array)):
        return_on_weights.append(weights_array[i] * paper_val * (1+hpr_array[i]))
    return np.sum(return_on_weights)


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
    return df.groupby('Sector').apply(lambda x: x.nlargest(top_n, filter_var)).reset_index(drop=True)



# Exponential backoff: retry up to 5 times, with exponential wait


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),  # Adjust backoff strategy here
    stop=stop_after_attempt(5),                         # Try max 5 times
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO)
)
def fetch_data_with_backoff(tickers: NDArray, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch Yf close data with backoff"""
    return yf.download(tickers.tolist(), start=start_date, end=end_date, auto_adjust=True)['Close']

# Vectorized function to calculate Sharpe ratio for multiple tickers
def calculate_sharpe_ratio(tickers: NDArray, tbill: pd.DataFrame, start_date: datetime, end_date: datetime):
    """Calcuate sharpe with back off, see backoff function inside"""
    # Download stock data with backoff
    
    stock_data = fetch_data_with_backoff(tickers, start_date, end_date)

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
    annualized_sharpe = daily_sharpe_ratio * np.sqrt(252)

    return annualized_sharpe

def get_stock_info(stocks: list, info_to_get: list) -> pd.DataFrame:
    if len(stocks) > 100:
        return 'Please set up logging, before getting info'
    
    end_list = []  # List of dictionaries

    for stock in stocks:
        stock_info = yf.Ticker(stock).info
        stock_data = {val: stock_info.get(val) for val in vals_to_get}
        stock_data['Ticker'] = stock  # Add ticker symbol
        end_list.append(stock_data)  
        
    df = pd.DataFrame(end_list)
    
    df['exDividendDate'] = pd.to_datetime(df['exDividendDate'], unit='s')
    # Convert list of dictionaries into DataFrame
    return df


def find_lending_or_borrowing_portfolio(
    expected_return_of_risky: float, 
    expected_std_risky: float, 
    risk_free_rate: float, 
    benchmark_std: float,
    stocks = list,
    weights = list, 
    add_margin: bool = True,
    risk_free_proxy: str = 'SGOV'
) -> dict:
    """
    Determines the optimal portfolio allocation between a risky asset and a risk-free asset 
    based on the given expected return, standard deviation, and benchmark standard deviation.
    """
    
    if expected_std_risky > benchmark_std:
        # Lending portfolio (T-bill added)
        risky_weight = benchmark_std / expected_std_risky
        portfolio_return = (1 - risky_weight) * risk_free_rate + risky_weight * expected_return_of_risky
        portfolio_std = risky_weight * expected_std_risky
        weights_adjusted = np.array(weights) * risky_weight
        
        return {
            "Risky Asset Weight": risky_weight,
            "Risk-Free Asset Weight": 1 - risky_weight,
            'Stocks': stocks + [risk_free_proxy],
            'Stock Weights': np.append(weights_adjusted, (1 - risky_weight)),
            "Expected Portfolio Return": portfolio_return,
            "Expected Portfolio Standard Deviation": portfolio_std,
            "Note": "Portfolio includes T-bills (expected std > benchmark std)."
        }

    if add_margin:
        # Borrowing (margin) portfolio
        risky_weight = benchmark_std / expected_std_risky
        portfolio_return = risky_weight * expected_return_of_risky
        portfolio_std = risky_weight * expected_std_risky
        margin_weights = np.array(weights) * risky_weight
        
        return {
            "Risky Asset Weight": risky_weight,
            "Risk-Free Asset Weight": "None: Portfolio remains the same, this portfolio std was below Benchmark.",
            'Stocks': stocks,
            'Stock Weights': margin_weights,
            "Expected Portfolio Return": portfolio_return,
            "Expected Portfolio Standard Deviation": portfolio_std,
            'Note': 'This is a Margin Portfolio'
        }

    return {
            "Risky Asset Weight": risky_weight,
            "Risk-Free Asset Weight": "None: Portfolio remains the same, this portfolio std was below Benchmark.",
            'Stocks': stocks,
            'Stock Weights': weights,
            "Expected Portfolio Return": portfolio_return,
            "Expected Portfolio Standard Deviation": portfolio_std,
            'Note': 'This is the orginal tanency portfolio'
        }

def plot_cum_ret(tickers: list,  weights: list, start_date: str, end_date: str, benchmark='SPY'):
    """ Plot cumaltive returns"""
    tickers += [benchmark]
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()

    # Calculate portfolio returns (weighted sum of asset returns)
    portfolio_returns = (returns.drop(columns=[benchmark]) * weights).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    
    # Plot individual asset cumulative returns
    for ticker in returns.columns:
        if ticker == benchmark:
            # Highlight SPY with a thicker line and distinct color
            cumulative_returns_ticker = (1 + returns[ticker]).cumprod() - 1
            cumulative_returns_ticker.plot(label=ticker, linewidth=2.5, color='darkblue')
        else:
            cumulative_returns_ticker = (1 + returns[ticker]).cumprod() - 1
            cumulative_returns_ticker.plot(label=ticker, linewidth=1, linestyle='dashed')

    # Plot portfolio cumulative returns
    cumulative_returns.plot(label="Portfolio", linewidth=2.5, color='green')

    # Add horizontal line at 0 for reference
    plt.axhline(0, color='black', linestyle='--', linewidth=3)

    # Formatting
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.title("Stock and Portfolio Cumulative Returns Over Time")
    plt.legend()
    plt.grid()
    plt.show()

print('\n---------------------------------')
print('finance_utils.py successfully loaded, updated last April. 29 2025 4:55')
print('---------------------------------')
print('\n')

# python3 utils/finance_utils.py