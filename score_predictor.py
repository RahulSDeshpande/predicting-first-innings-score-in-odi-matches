def custom_accuracy(y_test,y_pred):
    right = 0

    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= 20):
            right += 1
    print((right/l)*100)


import pandas as pd
# Importing the dataset
dataset = pd.read_csv('odi.csv')
X = dataset.iloc[:,[8,9,10,11,12,13,14,15,16]].values
y = dataset.iloc[:, 17].values


# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)

y_pred = lin.predict(X_test)
print(lin.score(X_test,y_test)*100)

custom_accuracy(y_test,y_pred)

import numpy as np
new_prediction = lin.predict(sc.transform(np.array([[66,0,4.3,14.67,66,0,14.67,36,28]])))
print(new_prediction)


    

    
    
							 


