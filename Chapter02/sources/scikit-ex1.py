import numpy as np 

from sklearn.linear_model import LinearRegression

# X is a matrix that represents the training dataset

# y is a vector of weights, to be associated with input dataset

X = np.array([[3], [5], [7], [9], [11]]).reshape(-1, 1) 

y = [8.0, 9.1, 10.3, 11.4, 12.6]  


lreg_model = LinearRegression()  


lreg_model.fit(X, y) 


# New data (unseen before)

new_data = np.array([[13]]) 


print('Model Prediction for new data: $%.2f' %  lreg_model.predict(new_data)[0]  ) 

