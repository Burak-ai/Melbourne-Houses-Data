import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("melb_data.csv")
#               na = not available
melbourne_data = df.dropna(axis=0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]


# print(melbourne_data.describe())

#                                   same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(X, y)

print(X.describe())
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


