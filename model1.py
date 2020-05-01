import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation


df=pd.read_csv("cardio_train.csv",sep=";")

df.drop(axis=1,columns=["id"],inplace=True)

df[(df<0)==True].any()

from scipy import stats
z=np.abs(stats.zscore(df))

df=df[(z<2).all(axis=1)]

#lets eliminate the negetive's of ap_hi and ap_lo

df=df.drop([4607,16021,20536,23988,35040])

df=df.drop([60106])

#Lets now try and scale the dataset
from sklearn.preprocessing import StandardScaler
std=StandardScaler()

y=df.iloc[:,-1:].values

df=df.drop(["cardio"],axis=1)

df=std.fit_transform(df)

#Now see the columns which are important

df=pd.DataFrame(df)

#Now droping the column of cholesterol

df=df.drop([6],axis=1)

#Now lets check the output variable Distribution

Y=pd.DataFrame(y)

#Quite distributed.... nice

#Now lets go for training..... and before that train test split

x=df.iloc[:,:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Now applying Classification Algorithm


"""# ANN"""

x_train.shape

model=Sequential()
model.add(Dense(12,input_dim=10,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(2,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=20)

pred=model.predict(x_test)>0.3

pred=pred.round()

# Saving model to disk
pickle.dump(model, open('cardiak.pkl','wb'))



