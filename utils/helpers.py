import pandas as pd


def filter_stocks(df: pd.DataFrame, 
                 min_market_cap: float = 100_000_000_000,  # 100B
                 min_profit_margin: float = 0.2,           # 20%
                 min_sharpe: float = 0.75) -> pd.DataFrame:
    
    # Handle missing values
    filtered_df = df.dropna(subset=['marketCap', 'profitMargins', 'sharpe_ratio'])
    
    # Create filter conditions
    market_cap_filter = filtered_df['marketCap'] > min_market_cap
    profit_margin_filter = filtered_df['profitMargins'] > min_profit_margin
    sharpe_filter = filtered_df['sharpe_ratio'] > min_sharpe
    
    # Apply filters
    filtered_df = filtered_df[
        market_cap_filter & 
        profit_margin_filter & 
        sharpe_filter
    ]
    
    # Add readable market cap column
    filtered_df['marketCap_B'] = filtered_df['marketCap'] / 1_000_000_000
    
    # Sort by market cap
    return filtered_df.sort_values('marketCap', ascending=False)

# Apply filters
