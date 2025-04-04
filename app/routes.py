from flask import Blueprint, request
from app.model import train_model_logic, predict_sales_logic

api_bp = Blueprint('api', __name__)

@api_bp.route('/train_model', methods=['POST'])
def train_model():
    return train_model_logic(request)

@api_bp.route('/predict_sales', methods=['POST'])
def predict_sales():
    return predict_sales_logic(request)
