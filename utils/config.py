
PROGRAM_START_DATE = '2024-04-15' # used to begin progam date
PROGRAM_END_DATE = '2025-04-15' # used to end program date

N_STOCKS_TO_GET = 100 # Used to select stocks from yf
TOP_N_STOCKS = 8 # Used to select top n stocks based on filtered critieria

print('Updated on 06/05/2025 5:56')
# python3 utils/config.py


# New Entry from portfolio_selection.ipynb
mpt_stocks_for_sharpe = ['AU', 'CWAN', 'CALM', 'HALO', 'KGC', 'ANYYY', 'MLI', 'EPRT']
mpt_stocks_for_profit = ['AU', 'CWAN', 'CALM', 'HALO', 'KGC', 'ANYYY', 'MLI', 'EPRT']

# New Entry from backtesting.ipynb
chosen_portfolio = {'Risky Asset Weight': 0.9717743686267484, 'Risk-Free Asset Weight': 0.028225631373251625, 'Stocks': ['AU', 'CWAN', 'CALM', 'HALO', 'KGC', 'ANYYY', 'MLI', 'EPRT', 'SGOV', 'SPY'], 'Stock Weights': [0.20885879727839787, 0.01017133851852177, 0.2505440598375792, 0.03925098631052384, 0.10072504421719634, 0.11236917647559652, 0.23589723907458876, 0.013957726914344012, 0.028225631373251625], 'Expected Portfolio Return': 0.5127199660391464, 'Expected Portfolio Standard Deviation': 0.19438517504703082, 'Note': 'Portfolio includes T-bills (expected std > benchmark std).'}
