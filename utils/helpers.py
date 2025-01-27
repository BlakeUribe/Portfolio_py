
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

print('\n---------------------------------')
print('helpers.py successfully loaded, updated last Jan. 15 2025')
print('---------------------------------')
print('\n')