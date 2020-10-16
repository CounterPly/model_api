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
            "version": "1.0",
            "description": "OpenAPI 3.0 Specification for an Amazon EC2 Hosted ML Model",

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
    from flask import request
    return f"Please refer to the <a href=\"{request.url_root}apidocs\">API documentation</a> for more details."


@app.route('/predict', methods=["GET"])
def predict():
    """Enter feature data.
    Input sample feature data to receive a prediction from the model.
    ---
    tags:
      - name: Single Prediction
    parameters:
      - name: building
        in: query
        type: string
        required: true
        description: Name of a Building on the Main NIH Campus that is the location of work for an employee.
      - name: num_publications
        in: query
        type: number
        required: true
        description: The total amount of peer-reviewed publications held by an employee.
      - name: num_conferences
        in: query
        type: number
        required: true
        description: The total amount of NIH-sanctioned conferences attended by an employee.
      - name: status
        in: query
        type: string
        required: true
        description: The classification of an employee as either "Parttime" or "Fulltime" (40 hours/week).
      - name: title
        in: query
        type: string
        required: true
        description: The official salutation of an employee; may correspond to educational level.
      - name: expertise
        in: query
        type: string
        required: true
        description: The specific field of expertise held by an employee.
      - name: institute
        in: query
        type: string
        required: true
        description: One of the 27 Institutes of the National Institutes of Health where the employee works.
      - name: num_postdocs
        in: query
        type: number
        required: true
        description: The total amount of Postdoctoral Fellows that an employee oversees.
      - name: num_reports
        in: query
        type: number
        required: true
        description: The total amount of individuals (contractors and Federal workers) overseen by the employee.
    responses:
        200:
            description: The output values of the request
    """
    # All Parameters
    building = request.args.get("building")
    num_publications = request.args.get("num_publications")
    num_conferences = request.args.get("num_conferences")
    status = request.args.get("status")
    title = request.args.get("title")
    expertise = request.args.get("expertise")
    institute = request.args.get("institute")
    num_postdocs = request.args.get("num_postdocs")
    num_reports = request.args.get("num_reports")

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
    app.run(host='0.0.0.0', port=5000)
