import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

independent_values = df[['Weight', 'Volume']]
dependent_values = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(independent_values.values, dependent_values.values)

# predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
print(regr.coef_)

# regr.predict([[3300, 1300]]) diye cagırınca [114.75968007] donuyor
# 107.2087328 + (1000 * 0.00755095) = 114.75968
