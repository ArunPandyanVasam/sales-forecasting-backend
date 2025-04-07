import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # 👈 swapped here
from app.utils import encoder, encode_event, extract_time_features
from flask import jsonify
from datetime import datetime

# Global model object
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 👈 swapped here

def train_model_logic(request):
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.json'):
        data = pd.read_json(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    required_columns = {"product", "historical_sales", "current_stock", "upcoming_event", "order_date"}
    if not required_columns.issubset(data.columns):
        return jsonify({"error": "Invalid file format. Required columns: product, historical_sales, current_stock, upcoming_event, order_date"}), 400

    if isinstance(data['historical_sales'].iloc[0], str):
        data['historical_sales'] = data['historical_sales'].apply(lambda x: list(map(int, x.split(','))))
    elif isinstance(data['historical_sales'].iloc[0], int):
        data['historical_sales'] = data['historical_sales'].apply(lambda x: [x])

    # Extract time features from order_date
    time_features_df = data['order_date'].apply(lambda x: extract_time_features(x))
    time_features = pd.DataFrame(time_features_df.tolist())

    encoder.fit(data[['upcoming_event']])
    encoded_events = encoder.transform(data[['upcoming_event']]).toarray()

    X_train, y_train = [], []
    for idx, row in data.iterrows():
        sales = row['historical_sales']
        for i in range(len(sales) - 1):
            time_row = time_features.iloc[idx].values
            encoded_event = encoded_events[idx]
            X_train.append([sales[i], row['current_stock']] + list(time_row) + list(encoded_event))
            y_train.append(sales[i + 1])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model.fit(X_train, y_train)
    return jsonify({"message": "Model trained successfully!"})

def predict_sales_logic(request):
    data = request.json
    product = data.get("product")
    last_sales = data.get("historical_sales")
    current_stock = data.get("current_stock", 0)
    upcoming_event = data.get("upcoming_event")
    order_date = data.get("order_date", datetime.today().strftime('%Y-%m-%d'))

    if not product or not last_sales:
        return jsonify({"error": "Missing required fields"}), 400

    encoded_input = encode_event(upcoming_event)
    time_input = extract_time_features(order_date)
    X_test = np.hstack(([last_sales[-1], current_stock], list(time_input), encoded_input))

    predicted_sales = model.predict([X_test])[0]
    reorder_suggestion = max(0, int(predicted_sales - current_stock))

    return jsonify({
        "product": product,
        "predicted_sales_next_month": int(predicted_sales),
        "reorder_suggestion": reorder_suggestion,
        "reasoning": "Predicted sales based on historical trends, time features, and events."
    })
