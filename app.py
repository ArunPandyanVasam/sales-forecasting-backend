from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Sample model (will be trained dynamically when data is received)
model = LinearRegression()
encoder = OneHotEncoder(handle_unknown='ignore')

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Train the AI model using historical sales data from CSV or JSON.
    """
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Detect file format
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.json'):
        data = pd.read_json(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    # Ensure correct columns exist
    required_columns = {"product", "historical_sales", "current_stock", "weather_forecast", "upcoming_event"}
    if not required_columns.issubset(data.columns):
        return jsonify({
            "error": "Invalid file format. Required columns: product, historical_sales, current_stock, weather_forecast, upcoming_event"
        }), 400

    # Check if historical_sales is a string (for CSV) and convert it to a list of integers
    if isinstance(data['historical_sales'].iloc[0], str):
        data['historical_sales'] = data['historical_sales'].apply(
            lambda x: list(map(int, x.split(','))) if isinstance(x, str) else x)

    # OneHotEncode categorical features (weather & event)
    categorical_features = data[['weather_forecast', 'upcoming_event']]
    encoder.fit(categorical_features)
    encoded_features = encoder.transform(categorical_features).toarray()

    # Prepare training data
    X_train = []
    y_train = []

    for idx, row in data.iterrows():
        sales = row['historical_sales']
        for i in range(len(sales) - 1):  # Use previous sales to predict the next
            X_train.append([sales[i], row['current_stock']])
            y_train.append(sales[i + 1])

    # Add encoded categorical data
    X_train = np.array(X_train)
    X_train = np.hstack((X_train, encoded_features.repeat(len(sales)-1, axis=0)))
    y_train = np.array(y_train)

    model.fit(X_train, y_train)

    return jsonify({"message": "Model trained successfully!"})

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    """
    Predict future sales based on the trained AI model.
    """
    data = request.json
    product = data.get("product")
    last_sales = data.get("historical_sales")

    if not product or not last_sales:
        return jsonify({"error": "Missing required fields"}), 400

    # Ensure proper input formatting
    last_sales = np.array(last_sales).reshape(-1, 1)
    current_stock = np.array([[data.get("current_stock", 0)]])

    # Convert input to DataFrame with correct feature names before encoding
    input_data = pd.DataFrame([[data["weather_forecast"], data["upcoming_event"]]],
                              columns=["weather_forecast", "upcoming_event"])

    # Apply encoding
    encoded_input = encoder.transform(input_data).toarray()

    # Combine all inputs
    X_test = np.hstack((last_sales[-1].reshape(1, -1), current_stock, encoded_input))
    predicted_sales = model.predict(X_test)[0]

    # Suggest reorder quantity based on stock levels
    reorder_suggestion = max(0, int(predicted_sales - data.get("current_stock", 0)))

    return jsonify({
        "product": product,
        "predicted_sales_next_month": int(predicted_sales),
        "reorder_suggestion": reorder_suggestion,
        "reasoning": "Predicted sales based on historical trends, weather, and events."
    })

if __name__ == '__main__':
    app.run(debug=True)