import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from numpy.typing import NDArray

# If Updated remember to adjust date at bottom


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MPTOptimizer:
    """Class will run MPT, with customized iterations, methods will return graph plotting efficient frontier, and return wieghts"""
    def __init__(self, stocks: list, start_date: str, end_date: str, iterations: int = 100_000):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.iterations = iterations
        self.returns = []
        self.stds = []
        self.weights = []

        # Download stock data
        self.data = yf.download(stocks, start=start_date, end=end_date)["Close"]
        
        # Calculate log returns
        self.stocks_lr = np.log(1 + self.data.pct_change()).dropna()

    def portfolio_return(self, weights):
        """Calculates expected portfolio return"""
        return np.dot(self.stocks_lr.mean(), weights) * 252

    def portfolio_std(self, weights):
        """Calculates portfolio standard deviation (risk)"""
        return np.sqrt(np.dot(weights.T, np.dot(self.stocks_lr.cov(), weights)) * 252)

    def generate_weights(self):
        """Generates random portfolio weights"""
        rand_weights = np.random.random(len(self.stocks_lr.columns))
        return rand_weights / rand_weights.sum()

    def simulate_portfolios(self):
        """Simulates random portfolios and stores their returns, risks, and weights"""
        for _ in range(self.iterations):
            weights = self.generate_weights()
            port_return = self.portfolio_return(weights)
            port_std = self.portfolio_std(weights)

            self.returns.append(port_return)
            self.stds.append(port_std)
            self.weights.append(weights)

    def plot_efficient_frontier(self): # returns figure
        """Plots the efficient frontier"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.stds, self.returns, c="red", s=0.5, alpha=0.5)

        # Maximum return portfolio
        max_return_idx = np.argmax(self.returns)
        plt.scatter(self.stds[max_return_idx], self.returns[max_return_idx], c="green", s=50, label="Max Return")

        # Minimum variance portfolio
        min_std_idx = np.argmin(self.stds)
        plt.scatter(self.stds[min_std_idx], self.returns[min_std_idx], c="blue", s=50, label="Min Variance")

        plt.xlabel("Portfolio Risk (Std Dev)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.legend()
        plt.show()

    def find_optimal_weights(self) -> dict:
        """Finds the optimal weights for the highest return portfolio"""
        max_return = max(self.returns)
        max_return_idx = np.argmax(self.returns)
        optimal_weights = self.weights[max_return_idx]
        
        stock_weights_dict = {}
        for i in range(len(self.stocks)):
            stock_weights_dict[self.stocks[i]] = optimal_weights[i]
            
        print(f"Max Return: {max_return * 100:.2f}%")
        print(f"Corresponding Standard Deviation: {self.stds[max_return_idx]:.4f}")
        print(f"Optimal Weights: {stock_weights_dict}")

        return stock_weights_dict

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
        stock_data = yf.download(stock, start=start_date, end=end_date, auto_adjust=False)['Close']
        if stock_data.empty:
            print(f"Warning: No data for {stock}")
            hpr_list.append(0)
            continue
        hpr = (stock_data.iloc[-1] - stock_data.iloc[0]) / stock_data.iloc[0]
        hpr_list.append(hpr)

    # Convert to NumPy for efficient calculations
    hpr_array = np.array(hpr_list)
    weights_array = np.array(weights)

    return_on_weights = []

    # Compute final portfolio value
    for i in range(len(hpr_array)):
        return_on_weights.append(weights_array[i] * paper_val * (1+hpr_array[i]))
        
    final_value = np.sum(return_on_weights)
    
    return final_value


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
print('finance_utils.py successfully loaded, updated last Feb. 04 2025 1:01')
print('---------------------------------')
print('\n')