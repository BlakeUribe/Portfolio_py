import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class MonteCarloSim:
    """Class to run MPT with customized iterations, returns efficient frontier graph, and optimal weights, with intent of maximaizing sharpe"""
    
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
        return np.dot(self.stocks_lr.mean(), weights) * 252  # Annualized return, r_protfolio = sum(mu_i * w_i) * 252, or dot product of mu * w_i

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
    
    

print('\n---------------------------------')
print('finance_utils.py successfully loaded, updated last June. 15 2025 4:55')
print('---------------------------------')
print('\n')
# python3 utils/models.py