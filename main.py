#import all the imp modules

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
%matplotlib inline


#load the digist 
digits=load_digits()

#train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.3)


from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
model=LogisticRegression()
x_scaled=preprocessing.scale(x_train)

model.fit(x_scaled, y_train)
model.predict([digits.data[67]])
