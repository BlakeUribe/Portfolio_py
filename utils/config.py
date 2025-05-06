
PROGRAM_START_DATE = '2024-04-15' # used to begin progam date
PROGRAM_END_DATE = '2025-04-15' # used to end program date

TOP_N_STOCKS = 8 # Used to select top n stocks based on filtered critieria

print('Updated on 04/15/2025 5:56')
# python3 utils/config.py

# New Entry from portfolio_selection.ipynb
mpt_stocks_for_sharpe = ['EXEL', 'THC', 'FUTU', 'FFIV', 'FOXA', 'SNA', 'MLI', 'COKE']
mpt_stocks_for_profit = ['EXEL', 'THC', 'FUTU', 'FFIV', 'FOXA', 'SNA', 'MLI', 'COKE']

# New Entry from backtesting.ipynb
chosen_portfolio = {'Risky Asset Weight': 1.0318328923852735, 'Risk-Free Asset Weight': 'None: Portfolio remains the same, this portfolio std was below Benchmark.', 'Stocks': ['EXEL', 'THC', 'FUTU', 'FFIV', 'FOXA', 'SNA', 'MLI', 'COKE', 'SPY'], 'Stock Weights': [0.24571091510251686, 0.18162668816539185, 0.11251342103201238, 0.3987355971633065, 0.044866789415142785, 0.011951276066965305, 0.028623983018949483, 0.007804222420988425], 'Expected Portfolio Return': 0.49448369454450297, 'Expected Portfolio Standard Deviation': 0.19438513228153506, 'Note': 'This is a Margin Portfolio'}
