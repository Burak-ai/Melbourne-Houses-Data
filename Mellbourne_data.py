import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


df = pd.read_csv("melb_data.csv")
#               na = not available
melbourne_data = df.dropna(axis=0)


y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
#                                   same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)


predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


print(X.describe())
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

melbourne_model = DecisionTreeRegressor()
# Define
melbourne_model.fit(train_X, train_y)
# Fit
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

