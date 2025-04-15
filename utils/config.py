import numpy as np

PROGRAM_START_DATE = '2024-03-12' # used to begin progam date
PROGRAM_END_DATE = '2025-03-12' # used to end program date

TOP_N_STOCKS = 8 # Used to select top n stocks based on filtered critieria

print('Updated on 03/12/2025 12:30')
# python3 utils/config.py


# New Entry from portfolio_selection.ipynb
mpt_stocks_for_sharpe = ['DE', 'NEM', 'CQP', 'GRMN', 'CSCO', 'VST', 'RCL', 'MPLX']
mpt_stocks_for_profit = ['DE', 'NEM', 'CQP', 'BBVA', 'EQR', 'BKNG', 'AVB', 'AAPL']

# New Entry from backtesting.ipynb
chosen_portfolio = {'Risky Asset Weight': 0.8382894952657721, 'Risk-Free Asset Weight': 0.16171050473422788, 'Stocks': ['DE', 'NEM', 'CQP', 'GRMN', 'CSCO', 'VST', 'RCL', 'MPLX', 'SGOV'], 'Stock Weights': [0.015581233401949376, 0.007306133337610667, 0.10209469131664765, 0.1740786200572725, 0.3540262805274835, 0.053924304833807196, 0.10649307073932002, 0.02478516105168133, 0.16171050473422788], 'Expected Portfolio Return': 0.3102012919562827, 'Expected Portfolio Standard Deviation': 0.13413692573975414, 'Note': 'Portfolio includes T-bills (expected std > benchmark std).'}
