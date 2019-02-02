import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle


model_filename = 'cl_model.sav'
model_r2_filename = 'cl_model_r2.csv'

def load_data():

	print("Start loading, cleaning data ...")
	df_orig = pd.read_csv('cl_Resources/home_value_calc.csv')
	df_orig = df_orig[~(df_orig == -666666666.0).any(axis=1)]
	df_orig = df_orig[~(df_orig == 666668684).any(axis=1)]
	df = df_orig.drop(["Poverty Count", "commute time car", 'Zipcode', 'Unemployment', 'zip_code','latitude', 'longitude', 'city', 'state', 'county', 'Bachelor holders', 'pop_biz','pop_stem', 'pop_tech' ], axis=1)
	df["Population Density"] = df["Population"]/df["Land-Sq-Mi"]
	df = df.drop("Land-Sq-Mi" , axis=1)
	print("Finish loading data\n")

	print("Start scaling data ...")
	X = df.drop("median_home_value", axis=1)
	y = df["median_home_value"].values.reshape(-1, 1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
	print("Finish scaling data\n")
	return [[X_train, X_test, y_train, y_test], X.keys()]


    ### Only run this function when building the model
def build_model(model_r2_filename, model_filename):

	data = load_data()[0]
	X_train = data[0]
	X_test = data[1]
	y_train = data[2]
	y_test = data[3]

	X_scaler = StandardScaler().fit(X_train)
	y_scaler = StandardScaler().fit(y_train)

	X_train_scaled = X_scaler.transform(X_train)
	X_test_scaled = X_scaler.transform(X_test)

	y_train_scaled = y_scaler.transform(y_train)
	y_test_scaled = y_scaler.transform(y_test)

	print("Start building model, this may take a little while ...")
	rf = RandomForestRegressor(n_estimators=200)
	rf = rf.fit(X_train_scaled, y_train_scaled)

	r2 = rf.score(X_test_scaled, y_test_scaled)

	model_r2_df = pd.DataFrame(data={'r2': [r2]})
	model_r2_df.to_csv(model_r2_filename, index=False)
	
	# save the model
	pickle.dump(rf, open(model_filename, 'wb'))
	print("Finish building model\n")

build_model(model_r2_filename, model_filename)        