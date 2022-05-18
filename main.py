import math
from pyexpat import model
from bitarray import test
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from xarray import Dataset
plt.style.use('fivethirtyeight')

#Get the dataframe
df=web.DataReader("AAPL", data_source='yahoo', start='2012-05-15', end='2022-05-16')
#print(df)
#Visualise
plt.figure(figsize=(16,8))
plt.title('Close Price His')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
#plt.show()

#DataFrame with only Close column Create
data=df.filter(['Close'])

dataset=data.values

#Number of Rows to train
training_data_len=math.ceil(len(dataset)*0.8)
#Scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
#print(scaled_data)

#Create Training Dataset
train_data=scaled_data[0:training_data_len,:]
#Split
xtrain=[]
ytrain=[]
for i in range(60,len(train_data)):
    xtrain.append(train_data[i-60:i,0])
    ytrain.append(train_data[i,0])
    if i<=60:
        print(xtrain)
        print(ytrain)
        print()
#Convert x y train to numpy
xtrain,ytrain=np.array(xtrain),np.array(ytrain)
#Reshape the data
xtrain=np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))
#Build LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(xtrain.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#Compile
model.compile(optimizer='adam',loss='mean_squared_error')
#Train
model.fit(xtrain,ytrain,batch_size=1,epochs=1)
#Create Test dataset
test_data=scaled_data[training_data_len-60:,:]
#Create x y test
xtest=[]
ytest=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    xtest.append(test_data[i-60:i,0])
#Convert Data to numpy
xtest=np.array(xtest)
#Reshape
xtest=np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))    
#Predict Value
prediction=model.predict(xtest)
prediction=scaler.inverse_transform(prediction)
#Get the RMSE(root mean squared error)
rmse =np.sqrt(np.mean(((prediction- ytest)**2)))
print("RMSE=",rmse)
#Plot Data
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Prediction']=prediction
#Visualise
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Prediction']])
plt.legend(['Train','Val','Prediction'],loc='lower right')
plt.show()
#Show Valid and Predicted Prices
#print(valid)
#Get the Quote
apple_quote=web.DataReader("AAPL", data_source='yahoo', start='2012-05-15', end='2022-05-16')
newdf=apple_quote.filter(['Close'])
last60day=newdf[-60:].values
last60dayscaled=scaler.transform(last60day)
xtest1=[]
xtest1.append(last60dayscaled)
xtest1=np.array(xtest1)
xtest1=np.reshape(xtest1,(xtest1.shape[0],xtest1.shape[1],1))
predprice=model.predict(xtest1)
predprice=scaler.inverse_transform(predprice)
print("Predicted Price ",predprice)

#Get second Quote
apple_quote2=web.DataReader("AAPL", data_source='yahoo', start='2022-05-17', end='2022-05-17')
print("The Actual Price",apple_quote2['Close'][1])