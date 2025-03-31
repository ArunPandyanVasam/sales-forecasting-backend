from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

app = Flask(__name__)

# Initialize AI Model
model = LinearRegression()
is_model_trained = False  # Flag to track training status


@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Train the AI model using historical sales data from CSV or JSON.
    """
    global is_model_trained

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Detect file format
    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            data = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        return jsonify({"error": f"Error reading file: {str(e)}"}), 400

    # Ensure correct columns exist
    required_columns = {"product", "historical_sales", "current_stock", "weather_forecast", "upcoming_event"}
    if not required_columns.issubset(data.columns):
        return jsonify({
            "error": f"Invalid file format. Required columns: {', '.join(required_columns)}"
        }), 400

    # Convert historical sales data from string to numerical list
    try:
        data['historical_sales'] = data['historical_sales'].apply(
            lambda x: list(map(int, x.split(','))) if isinstance(x, str) else x)
    except ValueError:
        return jsonify({"error": "Invalid data format in 'historical_sales'"}), 400

    # Prepare training data
    X_train, y_train = [], []
    for _, row in data.iterrows():
        sales = row['historical_sales']
        if len(sales) < 2:
            continue  # Skip products with insufficient data
        for i in range(len(sales) - 1):
            X_train.append([sales[i]])
            y_train.append(sales[i + 1])

    if not X_train:
        return jsonify({"error": "Insufficient data to train the model"}), 400

    # Train the model
    model.fit(np.array(X_train).reshape(-1, 1), np.array(y_train))
    is_model_trained = True  # Mark model as trained

    return jsonify({"message": "Model trained successfully!"})


@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    """
    Predict future sales based on the trained AI model.
    """
    global is_model_trained

    if not is_model_trained:
        return jsonify({"error": "Model is not trained yet. Please train the model first."}), 400

    data = request.get_json()
    product = data.get("product")
    last_sales = data.get("historical_sales")
    current_stock = data.get("current_stock", 0)

    if not product or not last_sales:
        return jsonify({"error": "Missing required fields (product, historical_sales)"}), 400

    try:
        last_sales = np.array(last_sales, dtype=int).reshape(-1, 1)
    except ValueError:
        return jsonify({"error": "Invalid format for 'historical_sales'. Must be a list of numbers."}), 400

    if len(last_sales) < 1:
        return jsonify({"error": "Insufficient historical sales data"}), 400

    try:
        predicted_sales = model.predict(last_sales[-1].reshape(1, -1))[0]
    except NotFittedError:
        return jsonify({"error": "Model is not trained yet. Please train before predicting."}), 400

    # Suggest reorder quantity based on stock levels
    reorder_suggestion = max(0, int(predicted_sales - current_stock))

    return jsonify({
        "product": product,
        "predicted_sales_next_month": int(predicted_sales),
        "reorder_suggestion": reorder_suggestion,
        "reasoning": "Predicted sales based on historical trends using AI."
    })


if __name__ == '__main__':
    app.run(debug=True)
