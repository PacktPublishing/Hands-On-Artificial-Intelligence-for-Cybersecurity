import numpy as np

def predict(data, w):  
    return data.dot(w)


# w is the vector of weights
w = np.array([0.1, 0.2, 0.3]) 

# matrices as input datasets
data1 = np.array([0.3, 1.5, 2.8]) 

data2 = np.array([0.5, 0.4, 0.9]) 

data3 = np.array([2.3, 3.1, 0.5])


data_in = np.array([data1[0],data2[0],data3[0]]) 

print('Predicted value: $%.2f' %  predict(data_in, w) )

