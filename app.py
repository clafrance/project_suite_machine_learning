import pandas as pd
import numpy as np
import cl_model as model

import json
import os
import csv
from flask import Flask, jsonify, request, render_template, flash, redirect, url_for, Markup

from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model_filename = 'cl_model.sav'
model_r2_filename = 'cl_model_r2.csv'



@app.route('/', methods=['GET', 'POST'])
def index():
    #return render_template("index.html")
    app.logger.debug('index.html rendered')
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # read the file
            file = request.files['file']
            # read the filename
            filename = file.filename
            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # input_df = pd.read_csv(filepath) 
            # X_new = input_df.values.T.tolist()
            input = []
            with open(filepath) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    for cell in row:
                        cell = float(cell)
                        input.append(cell)
        
            X_new = [input]
            #model.make_prediction(X_new, model_r2_filename, model_filename)
            
            result = model.make_prediction(X_new, model_r2_filename, model_filename)
            return jsonify(result)
    return render_template("index.html", )



if __name__ == "__main__":
    app.run(debug=True)


