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



# This function will call the saved model to make prediction
# Example of paramater: X_new = [[29489, 44.2, 116741, 50410, 3755,	1345, 2.261860355, 2.665400658, 244, 51, 1113.044463]]	

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
	print("Finish making prediction\n")
	print(prediction[0])
	print(r2)
	#return {"Prediction": prediction[0], "R2": r2, "importance": importances_list}
	return {"Prediction": prediction[0], "R2": r2}

#Step1: Build the model and save it to be used later. Uncomment below line:
#build_model(model_r2_filename, model_filename)
#Step2: You can test out the model built by un-comment below two lines and run the test sample data:
#X_new = [[29489, 44.2, 116741, 50410, 3755,	1345, 2.261860355, 2.665400658, 244, 51, 1113.044463]]		
#make_prediction(X_new, model_r2_filename, model_filename)
