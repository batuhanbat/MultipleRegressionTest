import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

independent_values = df[['TUFE', 'USD', 'EUR', 'GBP', 'CHF', 'ALTIN']]
dependent_values = df['TURKIYE']

regr = linear_model.LinearRegression()
regr.fit(independent_values.values, dependent_values.values)

# predict the Turkiye konut price(TL/m2) where TUFE=1500, USD=27, EUR=28, GBP=30, CHF=29, ALTIN=1600
predictedTurkiye = regr.predict([[1500, 27, 28, 30, 29, 1600]])

print(predictedTurkiye)
