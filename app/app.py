# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from joblib import load

global model

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



@app.route('/')
def landing():
    from flask import request
    return f"Please refer to the <a href=\"{request.url_root}apidocs\">API documentation</a> for more details."


@app.route('/predict', methods=["GET"])
def send_json_to_requester():
    """Enter feature data.
    Press "Try it out" on the right side of the screen to test out sample data against the API and receive predictions.
    ---
    tags:
      - name: Test Prediction
    parameters:
      - name: building
        in: query
        type: string
        required: true
        description: Name of a Building on the Main NIH Campus that is the location of work for an employee.
        default: 'Building 10'
      - name: num_publications
        in: query
        type: number
        required: true
        description: The total amount of peer-reviewed publications held by an employee.
        default: 20
      - name: num_conferences
        in: query
        type: number
        required: true
        description: The total amount of NIH-sanctioned conferences attended by an employee.
        default: 30
      - name: status
        in: query
        type: string
        required: true
        description: The classification of an employee as either 'Parttime' or 'Fulltime' (40 hours/week).
        default: 'Fulltime'
      - name: title
        in: query
        type: string
        required: true
        description: The official salutation of an employee; may correspond to educational level.
        default: 'Dr'
      - name: expertise
        in: query
        type: string
        required: true
        description: The specific field of expertise held by an employee.
        default: 'Infectious Diseases'
      - name: institute
        in: query
        type: string
        required: true
        description: One of the 27 Institutes of the National Institutes of Health where the employee works.
        default: 'NHLBI'
      - name: num_postdocs
        in: query
        type: number
        required: true
        description: The total amount of Postdoctoral Fellows that an employee oversees.
        default: 3
      - name: num_reports
        in: query
        type: number
        required: true
        description: The total amount of individuals (contractors and Federal workers) overseen by the employee.
        default: 7
    responses:
        200:
            description: The output values of the request
    """
    # All parameters
    building = request.args.get("building")
    num_publications = request.args.get("num_publications")
    num_conferences = request.args.get("num_conferences")
    status = request.args.get("status")
    title = request.args.get("title")
    expertise = request.args.get("expertise")
    institute = request.args.get("institute")
    num_postdocs = request.args.get("num_postdocs")
    num_reports = request.args.get("num_reports")

    # collect the incoming data
    num_conferences = int(num_conferences)
    is_fulltime = 1 if status == 'Fulltime' else 0
    postdocs = int(num_postdocs)
    reports = int(num_reports)
    publications = int(num_publications)

    # one_hot_encode
    building_10 = 1
    building_14 = 0
    building_Unknown = 0
    building_nan = 0
    if building[0] == 'Building 10':
        building_10 = 1
    if building[0] == 'Building 14':
        building_14 = 1

    # expertise
    expertise_infectious_diseases = 0
    expertise_bioinformatics = 0
    expertise_nan = 0
    if expertise == 'Infectious Diseases':
        expertise_infectious_diseases = 0
    if expertise == 'Bioinformatics':
        expertise_bioinformatics = 0

    # title
    title_Dr = 0
    title_Miss = 0
    title_Mr = 0
    title_Mrs = 0
    title_Ms = 0
    title_Unknown = 0
    title_nan = 0
    if title == 'Dr':
        title_Dr = 1
    if title == 'Miss':
        title_Miss = 1
    if title == 'Mr':
        title_Mr = 1
    if title == 'Mrs':
        title_Mrs = 1
    if title == 'Ms':
        title_Ms = 1
    if title == 'Unknown':
        title_Unknown = 1

    # institute
    institute_nci = 0
    institute_nei = 0
    institute_nhlbi = 0
    institute_nhgri = 0
    institute_nia = 0
    institute_niaid = 0
    institute_nimh = 0
    institute_Unknown = 0
    institute_nan = 0
    if institute == 'NCI':
        institute_nci = 1
    if institute == 'NEI':
        institute_nei = 1
    if institute == 'NHLBI':
        institute_nhlbi = 1
    if institute == 'NHGRI':
        institute_nhgri = 1
    if institute == 'NIA':
        institute_nia = 1
    if institute == 'NIAID':
        institute_niaid = 1
    if institute == 'NIMH':
        institute_nimh = 1
    if institute == 'Unknown':
        institute_Unknown = 1

    # predict
    user_designed_employee = [[num_conferences, postdocs, reports, publications, is_fulltime,
                               expertise_infectious_diseases, expertise_bioinformatics, expertise_nan,
                               institute_nci, institute_nei, institute_nhlbi, institute_nhgri, institute_nia,
                               institute_niaid, institute_nimh, institute_Unknown, institute_nan, building_10,
                               building_14, building_Unknown, building_nan, title_Dr, title_Miss, title_Mr,
                               title_Mrs, title_Ms, title_Unknown, title_nan]]

    # model is already in memory from earlier
    y_pred = model.predict_proba(user_designed_employee)
    perc = y_pred[0][1] * 100
    return f"There\'s a {str(round(perc, 2))}% chance that the employee will work onsite."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
