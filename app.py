import os

import pandas as pd
import numpy as np
#import model.py as model


from flask import Flask, jsonify, render_template
from flask import request

app = Flask(__name__)


#################################################
# Database Setup
#################################################



@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index.html")


# @app.route("/names")
# def names():
#      """Return ???."""
#     #results = session.query(Passenger).all()

#     # Create a dictionary from the row data and append to a list of all_passengers
#     # all_passengers = []
#     # for passenger in results:
#     #     passenger_dict = {}
#     #     passenger_dict["name"] = passenger.name
#     #     passenger_dict["age"] = passenger.age
#     #     passenger_dict["sex"] = passenger.sex
#     #     all_passengers.append(passenger_dict)

#     # return jsonify(all_passengers)



@app.route('/addRegion', methods=['POST'])
def addRegion():
    return (request.form['htmlTAG'])




if __name__ == "__main__":
    app.run(debug=True)


