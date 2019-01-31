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
	df = df_orig.drop(["Poverty Count", "commute time car", 'Zipcode', 'zip_code','latitude', 'longitude', 'city', 'state', 'county', 'Bachelor holders', 'pop_biz','pop_stem' ], axis=1)
	df["Population Density"] = df["Population"]/df["Land-Sq-Mi"]
	print("Finish loading data\n")

	print("Start scaling data ...")
	X = df.drop("median_home_value", axis=1)
	y = df["median_home_value"].values.reshape(-1, 1)

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

	# X_scaler = StandardScaler().fit(X_train)
	# y_scaler = StandardScaler().fit(y_train)

	# X_train_scaled = X_scaler.transform(X_train)
	# X_test_scaled = X_scaler.transform(X_test)

	# y_train_scaled = y_scaler.transform(y_train)
	# y_test_scaled = y_scaler.transform(y_test)
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



# This function will call the saved model to make prediction
# Example of paramater: X_new = [[17423.0, 45.0, 56714.0, 30430.0, 1353.0, 975.0, 8.391207, 479.0, 2.749240, 149, 240, 49, 11.442, 1522.723300]]
def make_prediction(X_new, model_r2_filename, model_filename):

	data = load_data()
	X_train = data[0][0]
	X_test = data[0][1]
	y_train = data[0][2]
	y_test = data[0][3]

	X_scaler = StandardScaler().fit(X_train)
	y_scaler = StandardScaler().fit(y_train)

	X_keys = data[1]

	X_new_scaled = X_scaler.transform(X_new)

	print("Start making prediction ...")
	loaded_model = pickle.load(open(model_filename, 'rb'))
	prediction_scaled = loaded_model.predict(X_new_scaled)
	prediction = y_scaler.inverse_transform(prediction_scaled)

	model_r2_df = pd.read_csv(model_r2_filename)
	r2 = model_r2_df["r2"][0]

	importances = loaded_model.feature_importances_
	importances_list = sorted(zip(loaded_model.feature_importances_, X_keys), reverse=True)
	print("Finisn making prediction\n")

	return {"Prediction": prediction[0], "R2": r2, "importance": importances_list}
