import pandas as pd  
        
from sklearn import datasets


iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

iris_df.head()

iris_df.describe()

