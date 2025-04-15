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
    """Class to run MPT with customized iterations, returns efficient frontier graph, and optimal weights"""
    
    def __init__(self, stocks: list, start_date: str, end_date: str, iterations: int = 100_000):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.iterations = iterations
        self.returns = []
        self.stds = []
        self.weights = []
        self.sharpe_ratios = []
         
        # Download stock data, and log returns
        self.data = yf.download(stocks, start=start_date, end=end_date)["Close"]
        self.stocks_lr = np.log(1 + self.data.pct_change()).dropna() #lr_stock i = ln(1+p_1/p_0)

        self.spy_data = yf.download('SPY', start=start_date, end=end_date)["Close"]
        
        # Download risk-free rate (Trying out mean of Tbill rate over time period)
        risk_free_rate_data = yf.download('^IRX', start=start_date, end=end_date)['Close']
        self.risk_free_rate = risk_free_rate_data.mean() / 100

        # Gather Bench mark data, log returns, and std
        self.spy_data = yf.download('SPY', start=start_date, end=end_date)["Close"]
        self.spy_lr = np.log(1 + self.spy_data.pct_change()).dropna()  # Log returns
        self.spy_return = self.spy_lr.sum() # Annulized log
        self.spy_std = self.spy_lr.std() * np.sqrt(252) # std

    def portfolio_return(self, weights):
        """Calculates expected portfolio return"""
        return np.dot(self.stocks_lr.mean(), weights) * 252  # Annualized return, r_protfolio = sum(mu_i * w_i) * 252, or dot product

    def portfolio_std(self, weights):
        """Calculates portfolio standard deviation (risk)"""
        return np.sqrt(np.dot(weights.T, np.dot(self.stocks_lr.cov(), weights)) * 252)  # Annualized risk, std_portfolio = sqrt(w^T * cov_matrix * w * 252)

    def generate_weights(self):
        """Generates random portfolio weights"""
        rand_weights = np.random.random(len(self.stocks_lr.columns))
        return rand_weights / rand_weights.sum()  # Normalize weights to sum to 1

    def simulate_portfolios(self):
        """Simulates random portfolios and stores their returns, risks, and weights"""
        for _ in range(self.iterations):
            weights = self.generate_weights()
            port_return = self.portfolio_return(weights)
            port_std = self.portfolio_std(weights)

            self.returns.append(port_return)
            self.stds.append(port_std)
            self.weights.append(weights)

    def plot_efficient_frontier(self): 
        """Plots the efficient frontier and the capital allocation line"""
        
        # Convert lists to arrays for easy access
        returns = np.array(self.returns)
        stds = np.array(self.stds)
        risk_free_rate = np.array(self.risk_free_rate)
        # Calculate Sharpe ratios
        self.sharpe_ratios = (returns - risk_free_rate) / stds

        # Find Tangency Portfolio (highest Sharpe ratio)
        max_sharpe_idx = np.argmax(self.sharpe_ratios)
        tangency_std = stds[max_sharpe_idx]
        tangency_return = returns[max_sharpe_idx]

        # Generate the CAL line (Capital Allocation Line)
        cal_std_range = np.linspace(0.1, tangency_std * 2, 100)  # Extend the line beyond the tangency point
        cal_returns = risk_free_rate + (tangency_return - risk_free_rate) * (cal_std_range / tangency_std)

        # Plot Efficient Frontier
        plt.scatter(stds, returns, c=self.sharpe_ratios, cmap='viridis', s=0.5, alpha=0.5, label="Efficient Frontier")
        # plt.scatter(stds, returns, c='red', s=0.5, alpha=0.5, label="Efficient Frontier")

        # Highlight the Tangency Portfolio
        plt.scatter(tangency_std, tangency_return, c="purple", s=35.0, label="Tangency Portfolio")

        # Plot the CAL Line (Capital Allocation Line)
        plt.plot(cal_std_range, cal_returns, color="black", linestyle="--", label="Capital Allocation Line (CAL)")
        
        # Plot benchmark if needed
        plt.scatter(self.spy_std, self.spy_return, c='red', s=35.0, label='SPY')
        # Labels and Legend
        plt.xlabel("Standard Deviation (Risk)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier and Capital Allocation Line")
        plt.legend()
        plt.grid(True)
        plt.show()

    def find_tangency_portfolio(self) -> dict:
        """Finds the optimal weights, and corresponding info for the highest return portfolio"""
        returns = np.array(self.returns)
        stds = np.array(self.stds)

        # Find the index of the maximum return
        max_sharpe_idx = np.argmax(self.sharpe_ratios)

        # max_return_idx = np.argmax(self.returns)
        optimal_weights = self.weights[max_sharpe_idx]
        
        # Create a dictionary for stock weights
        stock_weights_dict = {self.stocks[i]: optimal_weights[i] for i in range(len(self.stocks))}
        tangecy_porfolio_dict = {'Max Sharpe': self.sharpe_ratios[max_sharpe_idx],
                                'Corresponding Return': returns[max_sharpe_idx],
                                'Corresponding Standard Deviation': stds[max_sharpe_idx]
        } 
        
        # Find and print Tangency Portfolio
        print(f'-- Optimal Portfolio (CAL) --')
        print(f"Max Sharpe Ratio: {round(self.sharpe_ratios[max_sharpe_idx], 2)}")
        print(f"Corresponding Return: {round(returns[max_sharpe_idx], 2)}")
        print(f"Corresponding Standard Deviation: {round(stds[max_sharpe_idx], 2)}")
        print(f"Optimal Weights: {stock_weights_dict}")


        return stock_weights_dict, tangecy_porfolio_dict


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
    return df.groupby('Sector').apply(lambda x: x.nlargest(top_n, filter_var)).reset_index(drop=True)


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
    # Download historical data
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
print('finance_utils.py successfully loaded, updated last March. 17 2025 7:32')
print('---------------------------------')
print('\n')

# python3 utils/finance_utils.py