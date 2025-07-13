
PROGRAM_START_DATE = '2024-07-12' # used to begin progam date
PROGRAM_END_DATE = '2025-07-12' # used to end program date

N_STOCKS_TO_GET = 3000 # Used to select stocks from yf
TOP_N_STOCKS = 8 # Used to select top n stocks based on filtered critieria

print('Updated on 06/12/2025 3:11', PROGRAM_END_DATE)
# python3 utils/config.py


# New Entry from portfolio_selection.ipynb
mpt_stocks_for_sharpe = ['LRN', 'MGY', 'KGC', 'ABT', 'ATGE', 'OMAB', 'FINV', 'IDCC']
mpt_stocks_for_profit = ['LRN', 'MGY', 'KGC', 'ABT', 'ATGE', 'OMAB', 'FINV', 'IDCC']

# New Entry from backtesting.ipynb
chosen_portfolio = {'Risky Asset Weight': 1.1610880015494847, 'Risk-Free Asset Weight': 'None: Portfolio remains the same, this portfolio std was below Benchmark.', 'Stocks': ['LRN', 'MGY', 'KGC', 'ABT', 'ATGE', 'OMAB', 'FINV', 'IDCC', 'SPY'], 'Stock Weights': [0.25485541723966326, 0.07546662363813973, 0.17716086292717315, 0.2451036768022216, 0.057666392031968085, 0.16509719898412803, 0.004720749744566888, 0.18101708018162393], 'Expected Portfolio Return': 0.6251713769332887, 'Expected Portfolio Standard Deviation': 0.20428190462991602, 'Note': 'This is a Margin Portfolio'}
