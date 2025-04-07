from sklearn.preprocessing import OneHotEncoder
import numpy as np
from datetime import datetime

# Global encoder instance
encoder = OneHotEncoder(handle_unknown='ignore')

def encode_event(event):
    """
    Encode the upcoming event using the pre-fitted encoder.
    Returns a 1D numpy array representing the encoded event.
    """
    try:
        encoded = encoder.transform([[event]]).toarray()
        return encoded[0]
    except Exception as e:
        print(f"Encoding error: {e}")
        return np.zeros(encoder.transform([['unknown']]).toarray().shape[1])

def extract_time_features(date_str):
    """
    Extract month, day of week, and season from order date.
    Format expected: 'YYYY-MM-DD'
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    month = date_obj.month
    day_of_week = date_obj.weekday()
    season = (month % 12 + 3) // 3  # Maps to 1-4 for season buckets
    return [month, day_of_week, season]
