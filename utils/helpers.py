import numpy as np
import pandas as pd

# Function to divide a list into chunks
def divide_chunks(l: list, n: int) -> list:
    """
    Divides a list into smaller chunks of size n.

    Args:
        l (list): The list to divide.
        n (int): The size of each chunk.

    Returns:
        Generator yielding chunks of the list.
    """
    for i in range(0, len(l), n): 
        yield l[i:i + n]
        
def separate_corr_pairs(corr_pairs: list, top_n: int) -> np.array:
    """
    Extracts and returns unique corralation pairs from a list of correlation pairs.

    Args:
        corr_pairs (list): List of stock correlation pairs in 'StockA_StockB' format.
        top_n (int): Number of unique stock symbols to return. Defaults to 8.

    Returns:
        np.array: Array of unique stock symbols from the correlation pairs.
    """
    pair_list = [pair.split('_') for pair in corr_pairs]  # Split stock pairs
    pair_list_1d = np.array(pair_list).flatten()  # Flatten into a 1D array
    unique_values = pd.unique(pair_list_1d)[:top_n]  # Get top unique stock symbols
    
    return unique_values

print('\n---------------------------------')
print('helpers.py successfully loaded, updated last Feb. 04 2025')
print('---------------------------------')
print('\n')