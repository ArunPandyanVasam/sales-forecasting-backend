from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
CORS(app)

# Initialize model and encoder
model = LinearRegression()
encoder = OneHotEncoder(handle_unknown='ignore')

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Train the model using the provided data (CSV/JSON).
    """
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.json'):
        data = pd.read_json(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    required_columns = {"product", "historical_sales", "current_stock", "upcoming_event"}
    if not required_columns.issubset(data.columns):
        return jsonify({"error": "Invalid file format. Required columns: product, historical_sales, current_stock, upcoming_event"}), 400

    # Ensure historical_sales is a list of integers
    if isinstance(data['historical_sales'].iloc[0], str):
        # If historical_sales are stored as a string, split by commas and convert to a list of integers
        data['historical_sales'] = data['historical_sales'].apply(lambda x: list(map(int, x.split(','))))
    elif isinstance(data['historical_sales'].iloc[0], int):
        # If it's a single integer, treat it as a list with one element
        data['historical_sales'] = data['historical_sales'].apply(lambda x: [x])

    # Encode categorical columns (events)
    categorical_features = data[['upcoming_event']]
    encoder.fit(categorical_features)
    encoded_features = encoder.transform(categorical_features).toarray()

    # Prepare the training data
    X_train, y_train = [], []
    for idx, row in data.iterrows():
        sales = row['historical_sales']
        for i in range(len(sales) - 1):
            X_train.append([sales[i], row['current_stock']])
            y_train.append(sales[i + 1])

    X_train = np.array(X_train)
    X_train = np.hstack((X_train, encoded_features.repeat(len(sales)-1, axis=0)))
    y_train = np.array(y_train)

    # Train the model
    model.fit(X_train, y_train)
    return jsonify({"message": "Model trained successfully!"})

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    """
    Predict future sales based on historical sales, current stock, and upcoming events.
    """
    data = request.json
    product = data.get("product")
    last_sales = data.get("historical_sales")
    if not product or not last_sales:
        return jsonify({"error": "Missing required fields"}), 400

    # Prepare input data
    last_sales = np.array(last_sales).reshape(-1, 1)
    current_stock = np.array([[data.get("current_stock", 0)]])
    input_data = pd.DataFrame([[data["upcoming_event"]]],
                              columns=["upcoming_event"])
    encoded_input = encoder.transform(input_data).toarray()

    # Prepare the test data for prediction
    X_test = np.hstack((last_sales[-1].reshape(1, -1), current_stock, encoded_input))
    predicted_sales = model.predict(X_test)[0]

    reorder_suggestion = max(0, int(predicted_sales - data.get("current_stock", 0)))
    return jsonify({
        "product": product,
        "predicted_sales_next_month": int(predicted_sales),
        "reorder_suggestion": reorder_suggestion,
        "reasoning": "Predicted sales based on historical trends and events."
    })

if __name__ == '__main__':
    app.run(debug=True)
