from flask import Flask, request, jsonify
from flask_cors import CORS
from model import train_model, predict_sales
from weather import get_weather

app = Flask(__name__)
CORS(app)

@app.route('/train_model', methods=['POST'])
def train():
    return train_model(request)

@app.route('/predict_sales', methods=['POST'])
def predict():
    return predict_sales(request, get_weather)

if __name__ == '__main__':
    app.run(debug=True)
