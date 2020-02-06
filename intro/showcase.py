import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#Data read
df = pd.read_csv("FuelConsumption.csv")

cdf = df[['ENGINESIZE', 'FUELCONSUMPTION_HWY']]

print(cdf.head)

#Spliting data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#Model deployment
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['FUELCONSUMPTION_HWY']])
regr.fit (train_x, train_y)

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['FUELCONSUMPTION_HWY']])
test_y_hat = regr.predict(test_x)

#Print scores
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

#Plot graph
plt.scatter(train['ENGINESIZE'], train['FUELCONSUMPTION_HWY'],  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("ENGINESIZE")
plt.ylabel('FUELCONSUMPTION_HWY')
plt.show()
