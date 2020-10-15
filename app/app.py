# -*- coding: utf-8 -*-

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app = Flask(__name__)

swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['swagger_ui_bundle_js'] = '//unpkg.com/swagger-ui-dist@3.3.0/swagger-ui-bundle.js'
swagger_config['swagger_ui_standalone_preset_js'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui-standalone-preset.js'
swagger_config['jquery_js'] = '//unpkg.com/jquery@2.2.4/dist/jquery.min.js'
swagger_config['swagger_ui_css'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui.css'


def create_swagger(app_):
    template = {
        "openapi": '3.0.0',
        "info": {
            "title": "Swagger API - Bentley Assignment",
            "description": "OpenAPI 3.0 Specification for an Amazon EC2 Hosted ML Model",
            "version": "1.0"
        },
        "basePath": "/",
        "schemes": [
            "http",
            "https"
        ]
    }
    return Swagger(app_, template=template)


create_swagger(app)

pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict', methods=["GET"])
def predict():
    """Enter feature data.
    Input sample feature data to receive a prediction from the model.
    ---
    tags:
      - name: Single Prediction
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    return str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_file(test_file):
    """ Upload CSV File.
    Upload a CSV File containing sample data to receive multiple predictions.
    ---
    tags:
      - name: Bulk Prediction
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    """
    df_test = pd.read_csv(request.files.get(test_file))
    print(df_test.head())
    prediction = model.predict(df_test)

    return str(list(prediction))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
