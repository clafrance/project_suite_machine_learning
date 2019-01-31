import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

model_filename = 'cl_model.sav'
model_description_filename = 'cl_model_description.csv'

### Only need to run this function when building the model
def build_model(model_description_filename, model_filename):
	print("Start loading, cleaning data ...")
	df = pd.read_csv('cl_Resources/census_data.csv')
	df_land = pd.read_csv('cl_Resources/Zipcode-Population-Density-2010.csv')
	df_unemployment = pd.read_csv('cl_Resources/Unemployment.csv')
	df.dropna(inplace=True)
	df_unemployment.dropna(inplace=True)

	# removes rows that has neg numbers
	df = df[~(df < 0).any(axis=1)]
	df = df.join(df_land.set_index('Zipcode'), on='Zipcode')
	df.dropna(inplace=True)
	df = df.join(df_unemployment.set_index('Zipcode'), on='Zipcode')
	df.dropna(inplace=True)
	df["Population Density"] = df["Population"]/df["Land-Sq-Mi"]
	df = df.drop("Poverty Count", axis=1)
	df.drop(columns=['Zipcode'], inplace=True)
	print("Finish loading data\n")

	print("Start scaling data ...")
	X = df[["Population", "Median Age", "Household Income", "Per Capita Income", "Poverty Rate", "Land-Sq-Mi", "Unemp Rate", "Population Density"]]
	y = df["median_home_value"].values.reshape(-1, 1)

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

	X_scaler = StandardScaler().fit(X_train)
	# y_scaler = StandardScaler().fit(y_train)

	X_train_scaled = X_scaler.transform(X_train)
	X_test_scaled = X_scaler.transform(X_test)

	# y_train_scaled = y_scaler.transform(y_train)
	# y_test_scaled = y_scaler.transform(y_test)
	y_train_scaled = y_train
	y_test_scaled = y_test
	print("Finish scaling data\n")

	print("Start building model, this may take a little while ...")
	rf = RandomForestRegressor(n_estimators=200)
	rf = rf.fit(X_train_scaled, y_train_scaled)
	r2 = rf.score(X_test_scaled, y_test_scaled)

	importances = rf.feature_importances_
	importances_list = sorted(zip(rf.feature_importances_, X.keys()), reverse=True)

	model_description_df = pd.DataFrame(data={'r2': [r2], 'importance': [importances_list]})
	model_description_df.to_csv(model_description_filename , index=False)

	# save the model
	pickle.dump(rf, open(model_filename, 'wb'))
	print("Finish building model\n")



# This function will call the model saved to make prediction
# Example of paramater: X_new = [[ 3.48460584, -0.82941627,  0.80809856,  0.34549878, -0.01890773, -0.33548015,  0.2246137 ,  1.35009033]]
def make_prediction(X_new, model_description_filename):

	print("Start making prediction ...")
	loaded_model = pickle.load(open(model_filename, 'rb'))
	# prediction_scaled = loaded_model.predict(X_new)
	# prediction = y_scaler.inverse_transform(prediction_scaled)
	prediction = loaded_model.predict(X_new)

	model_description_df = pd.read_csv(model_description_filename)
	r2 = model_description_df["r2"][0]
	print("Finisn making prediction\n")

	return {"Prediction": prediction[0], "R2": r2}



# def prediction(X_new):
# 	model_filename = 'cl_model.sav'

# 	print("Start loading, cleaning data ...")
# 	df = pd.read_csv('cl_Resources/census_data.csv')
# 	df_land = pd.read_csv('cl_Resources/Zipcode-Population-Density-2010.csv')
# 	df_unemployment = pd.read_csv('cl_Resources/Unemployment.csv')
# 	df.dropna(inplace=True)
# 	df_unemployment.dropna(inplace=True)

# 	# removes rows that has neg numbers
# 	df = df[~(df < 0).any(axis=1)]
# 	df = df.join(df_land.set_index('Zipcode'), on='Zipcode')
# 	df.dropna(inplace=True)
# 	df = df.join(df_unemployment.set_index('Zipcode'), on='Zipcode')
# 	df.dropna(inplace=True)
# 	df["Population Density"] = df["Population"]/df["Land-Sq-Mi"]
# 	df = df.drop("Poverty Count", axis=1)
# 	df.drop(columns=['Zipcode'], inplace=True)
# 	print("Finish loading data\n")


# 	print("Start scaling data ...")
# 	X = df[["Population", "Median Age", "Household Income", "Per Capita Income", "Poverty Rate", "Land-Sq-Mi", "Unemp Rate", "Population Density"]]
# 	y = df["median_home_value"].values.reshape(-1, 1)

# 	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 	X_scaler = StandardScaler().fit(X_train)
# 	y_scaler = StandardScaler().fit(y_train)

# 	X_train_scaled = X_scaler.transform(X_train)
# 	X_test_scaled = X_scaler.transform(X_test)

# 	y_train_scaled = y_scaler.transform(y_train)
# 	y_test_scaled = y_scaler.transform(y_test)
# 	print("Finish scaling data\n")

	
# 	print("Start building model, it may take a little while ...")
# 	rf = RandomForestRegressor(n_estimators=200)
# 	rf = rf.fit(X_train_scaled, y_train_scaled)
# 	r2 = rf.score(X_test_scaled, y_test_scaled)

# 	importances = rf.feature_importances_
# 	importances_list = sorted(zip(rf.feature_importances_, X.keys()), reverse=True)

# 	# save the model
# 	pickle.dump(rf, open(model_filename, 'wb'))
# 	print("Finish building model\n")


# 	print("Start making prediction ...")
# 	loaded_model = pickle.load(open(model_filename, 'rb'))
# 	prediction_scaled = loaded_model.predict(X_new)
# 	prediction = y_scaler.inverse_transform(prediction_scaled)
# 	print("Finisn making prediction\n")

# 	return (prediction, importances_list, r2)
