from sklearn.preprocessing import OneHotEncoder
import numpy as np

encoder = OneHotEncoder(handle_unknown='ignore')

def encode_event(event):
    encoded = encoder.transform([[event]]).toarray()
    return encoded[0]  # return 1D array
