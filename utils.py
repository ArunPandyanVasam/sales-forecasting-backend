import pandas as pd

def preprocess_historical_sales(data):
    """
    Preprocess historical sales to ensure it's in the correct format.
    """
    if isinstance(data['historical_sales'].iloc[0], str):
        # If historical_sales are stored as a string, split by commas and convert to a list of integers
        data['historical_sales'] = data['historical_sales'].apply(lambda x: list(map(int, x.split(','))))
    elif isinstance(data['historical_sales'].iloc[0], int):
        # If it's a single integer, treat it as a list with one element
        data['historical_sales'] = data['historical_sales'].apply(lambda x: [x])
    return data
