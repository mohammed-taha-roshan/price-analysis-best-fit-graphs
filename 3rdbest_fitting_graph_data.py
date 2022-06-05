# 3  sqft_living15
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
dff=pd.read_csv(file_name)
dff=dff.drop("id",axis=1)
dff=dff.drop("Unnamed: 0",axis=1)


dff=dff.replace('?',np.nan)
avg1=dff['sqft_living15'].astype('float').mean(axis=0)
dff=dff.replace(np.nan,avg1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dff,dff,test_size=.2,random_state=4);


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(X_train[['sqft_living15']])
train_y = np.asanyarray(X_train[['price']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


plt.scatter(X_train.sqft_living15, X_train.price,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("sqft_living15")
plt.ylabel("price")
plt.show()
print("****************************")
test_x = np.asanyarray(X_test[['sqft_living15']])
test_y = np.asanyarray(X_test[['price']])
test_y_ = regr.predict(test_x)
mae= np.mean(np.absolute(test_y_ - test_y))
mae_=mae/21613
mse=np.mean((test_y_ - test_y) ** 2)
mse_=mse/21613
print("Mean absolute error: " , mae_)
print("Residual sum of squares (MSE): " ,mse_ )